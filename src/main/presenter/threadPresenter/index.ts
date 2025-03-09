import {
  IThreadPresenter,
  CONVERSATION,
  CONVERSATION_SETTINGS,
  MESSAGE_ROLE,
  MESSAGE_STATUS,
  MESSAGE_METADATA,
  SearchResult,
  MODEL_META
} from '../../../shared/presenter'
import { ISQLitePresenter } from '../../../shared/presenter'
import { MessageManager } from './messageManager'
import { ILlmProviderPresenter } from '../../../shared/presenter'
import { eventBus } from '@/eventbus'
import {
  AssistantMessage,
  Message,
  AssistantMessageBlock,
  SearchEngineTemplate
} from '@shared/chat'
import { approximateTokenSize } from 'tokenx'
import { getModelConfig } from '../llmProviderPresenter/modelConfigs'
import { SearchManager } from './searchManager'
import { getArtifactsPrompt } from '../llmProviderPresenter/promptUtils'
import { getFileContext } from './fileContext'
import { ContentEnricher } from './contentEnricher'
import { CONVERSATION_EVENTS, STREAM_EVENTS } from '@/events'

const DEFAULT_SETTINGS: CONVERSATION_SETTINGS = {
  systemPrompt: '',
  temperature: 0.7,
  contextLength: 1000,
  maxTokens: 2000,
  providerId: 'openai',
  modelId: 'gpt-4',
  artifacts: 0
}

interface GeneratingMessageState {
  message: AssistantMessage
  conversationId: string
  startTime: number
  firstTokenTime: number | null
  promptTokens: number
  reasoningStartTime: number | null
  reasoningEndTime: number | null
  lastReasoningTime: number | null
}
const SEARCH_PROMPT_TEMPLATE = `You are an expert in organizing search results.Write an accurate answer concisely for a given question, citing the search results as needed. Your answer must be correct, high-quality, and written by an expert using an unbiased and journalistic tone. Your answer must be written in the same language as the question, even if language preference is different. Cite search results using [index] at the end of sentences when needed, for example "Ice is less dense than water.[1][2]" NO SPACE between the last word and the citation. Cite the most relevant results that answer the question. Avoid citing irrelevant results. Write only the response. Use markdown for formatting.

- Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
- Use markdown code blocks to write code, including the language for syntax highlighting.
- Use LaTeX to wrap ALL math expression. Always use double dollar signs $$, for example $$x^4 = x - 3$$.
- DO NOT include any URL's, only include citations with numbers, eg [1].
- DO NOT include references (URL's at the end, sources).
- Use footnote citations at the end of applicable sentences(e.g, [1][2]).
- Write more than 100 words (2 paragraphs).
- In the response avoid referencing the citation directly
- Print just the response text.
<search_results>
{{SEARCH_RESULTS}}
</search_results>
<user_query>
{{USER_QUERY}}
</user_query>
  `
// 格式化搜索结果的函数
export function formatSearchResults(results: SearchResult[]): string {
  return results
    .map(
      (result, index) => `source ${index + 1}：${result.title}
URL: ${result.url}
content：${result.content || ''}
---`
    )
    .join('\n\n')
}
// 生成带搜索结果的提示词
export function generateSearchPrompt(query: string, results: SearchResult[]): string {
  return SEARCH_PROMPT_TEMPLATE.replace('{{SEARCH_RESULTS}}', formatSearchResults(results)).replace(
    '{{USER_QUERY}}',
    query
  )
}

export class ThreadPresenter implements IThreadPresenter {
  private activeConversationId: string | null = null
  private sqlitePresenter: ISQLitePresenter
  private messageManager: MessageManager
  private llmProviderPresenter: ILlmProviderPresenter
  private searchManager: SearchManager
  private generatingMessages: Map<string, GeneratingMessageState> = new Map()
  private searchAssistantModel: MODEL_META | null = null
  private searchAssistantProviderId: string | null = null

  constructor(sqlitePresenter: ISQLitePresenter, llmProviderPresenter: ILlmProviderPresenter) {
    this.sqlitePresenter = sqlitePresenter
    this.messageManager = new MessageManager(sqlitePresenter)
    this.llmProviderPresenter = llmProviderPresenter
    this.searchManager = new SearchManager()

    // 初始化时处理所有未完成的消息
    this.initializeUnfinishedMessages()

    eventBus.on(STREAM_EVENTS.RESPONSE, async (msg) => {
      const { eventId, content, reasoning_content } = msg
      const state = this.generatingMessages.get(eventId)
      if (state) {
        // 记录第一个token的时间
        if (state.firstTokenTime === null && (content || reasoning_content)) {
          state.firstTokenTime = Date.now()
          await this.messageManager.updateMessageMetadata(eventId, {
            firstTokenTime: Date.now() - state.startTime
          })
        }

        // 处理reasoning_content的时间戳
        if (reasoning_content) {
          if (state.reasoningStartTime === null) {
            state.reasoningStartTime = Date.now()
            await this.messageManager.updateMessageMetadata(eventId, {
              reasoningStartTime: Date.now() - state.startTime
            })
          }
          state.lastReasoningTime = Date.now()
        }

        const lastBlock = state.message.content[state.message.content.length - 1]
        if (content) {
          if (lastBlock && lastBlock.type === 'content') {
            lastBlock.content += content
          } else {
            if (lastBlock) {
              lastBlock.status = 'success'
            }
            state.message.content.push({
              type: 'content',
              content: content,
              status: 'loading',
              timestamp: Date.now()
            })
          }
        }
        if (reasoning_content) {
          if (lastBlock && lastBlock.type === 'reasoning_content') {
            lastBlock.content += reasoning_content
          } else {
            if (lastBlock) {
              lastBlock.status = 'success'
            }
            state.message.content.push({
              type: 'reasoning_content',
              content: reasoning_content,
              status: 'loading',
              timestamp: Date.now()
            })
          }
        }
      }
    })
    eventBus.on(STREAM_EVENTS.END, async (msg) => {
      const { eventId } = msg
      const state = this.generatingMessages.get(eventId)
      if (state) {
        state.message.content.forEach((block) => {
          block.status = 'success'
        })

        // 计算completion tokens
        let completionTokens = 0
        for (const block of state.message.content) {
          if (block.type === 'content' || block.type === 'reasoning_content') {
            completionTokens += approximateTokenSize(block.content)
          }
        }

        // 检查是否有内容块
        const hasContentBlock = state.message.content.some(
          (block) => block.type === 'content' || block.type === 'reasoning_content'
        )

        // 如果没有内容块，添加错误信息
        if (!hasContentBlock) {
          state.message.content.push({
            type: 'error',
            content: 'common.error.noModelResponse',
            status: 'error',
            timestamp: Date.now()
          })
        }

        const totalTokens = state.promptTokens + completionTokens
        const generationTime = Date.now() - (state.firstTokenTime ?? state.startTime)
        const tokensPerSecond = completionTokens / (generationTime / 1000)

        // 如果有reasoning_content，记录结束时间
        const metadata: Partial<MESSAGE_METADATA> = {
          totalTokens,
          inputTokens: state.promptTokens,
          outputTokens: completionTokens,
          generationTime,
          firstTokenTime: state.firstTokenTime ? state.firstTokenTime - state.startTime : 0,
          tokensPerSecond
        }

        if (state.reasoningStartTime !== null && state.lastReasoningTime !== null) {
          metadata.reasoningStartTime = state.reasoningStartTime - state.startTime
          metadata.reasoningEndTime = state.lastReasoningTime - state.startTime
        }

        // 更新消息的usage信息
        await this.messageManager.updateMessageMetadata(eventId, metadata)

        await this.messageManager.updateMessageStatus(eventId, 'sent')
        await this.messageManager.editMessage(eventId, JSON.stringify(state.message.content))
        this.generatingMessages.delete(eventId)
      }
    })
    eventBus.on(STREAM_EVENTS.ERROR, async (msg) => {
      const { eventId, error } = msg
      const state = this.generatingMessages.get(eventId)
      if (state) {
        await this.handleMessageError(eventId, String(error))
        this.generatingMessages.delete(eventId)
      }
    })
  }
  setSearchAssistantModel(model: MODEL_META, providerId: string) {
    this.searchAssistantModel = model
    this.searchAssistantProviderId = providerId
  }
  getSearchEngines(): SearchEngineTemplate[] {
    return this.searchManager.getEngines()
  }
  getActiveSearchEngine(): SearchEngineTemplate {
    return this.searchManager.getActiveEngine()
  }
  setActiveSearchEngine(engineName: string) {
    this.searchManager.setActiveEngine(engineName)
  }

  /**
   * 处理消息错误状态的公共函数
   * @param messageId 消息ID
   * @param errorMessage 错误信息
   */
  private async handleMessageError(
    messageId: string,
    errorMessage: string = 'common.error.requestFailed'
  ): Promise<void> {
    const message = await this.messageManager.getMessage(messageId)
    if (!message) {
      return
    }

    let content: AssistantMessageBlock[] = []
    try {
      content = JSON.parse(message.content)
    } catch (e) {
      content = []
    }

    // 将所有loading状态的block改为error
    content.forEach((block: AssistantMessageBlock) => {
      if (block.status === 'loading') {
        block.status = 'error'
      }
    })

    // 添加错误信息block
    content.push({
      type: 'error',
      content: errorMessage,
      status: 'error',
      timestamp: Date.now()
    })

    // 更新消息状态和内容
    await this.messageManager.updateMessageStatus(messageId, 'error')
    await this.messageManager.editMessage(messageId, JSON.stringify(content))
  }

  /**
   * 初始化未完成的消息
   */
  private async initializeUnfinishedMessages(): Promise<void> {
    try {
      // 获取所有对话
      const { list: conversations } = await this.getConversationList(1, 1000)

      for (const conversation of conversations) {
        // 获取每个对话的消息
        const { list: messages } = await this.getMessages(conversation.id, 1, 1000)

        // 找出所有pending状态的assistant消息
        const pendingMessages = messages.filter(
          (msg) => msg.role === 'assistant' && msg.status === 'pending'
        )

        // 处理每个未完成的消息
        for (const message of pendingMessages) {
          await this.handleMessageError(message.id, 'common.error.sessionInterrupted')
        }
      }
    } catch (error) {
      console.error('初始化未完成消息失败:', error)
    }
  }

  async renameConversation(conversationId: string, title: string): Promise<CONVERSATION> {
    return await this.sqlitePresenter.renameConversation(conversationId, title)
  }

  async createConversation(
    title: string,
    settings: Partial<CONVERSATION_SETTINGS> = {}
  ): Promise<string> {
    const latestConversation = await this.getLatestConversation()

    if (latestConversation) {
      const { list: messages } = await this.getMessages(latestConversation.id, 1, 1)
      if (messages.length === 0) {
        await this.setActiveConversation(latestConversation.id)
        return latestConversation.id
      }
    }
    let defaultSettings = DEFAULT_SETTINGS
    if (latestConversation?.settings) {
      defaultSettings = { ...latestConversation.settings }
      defaultSettings.systemPrompt = ''
    }
    Object.keys(settings).forEach((key) => {
      if (settings[key] === undefined || settings[key] === null || settings[key] === '') {
        delete settings[key]
      }
    })
    const mergedSettings = { ...defaultSettings, ...settings }
    const defaultModelsSettings = getModelConfig(mergedSettings.modelId)
    if (defaultModelsSettings) {
      mergedSettings.maxTokens = defaultModelsSettings.maxTokens
      mergedSettings.contextLength = defaultModelsSettings.contextLength
      mergedSettings.temperature = defaultModelsSettings.temperature
    }
    const conversationId = await this.sqlitePresenter.createConversation(title, mergedSettings)
    await this.setActiveConversation(conversationId)
    return conversationId
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await this.sqlitePresenter.deleteConversation(conversationId)
    if (this.activeConversationId === conversationId) {
      this.activeConversationId = null
    }
  }

  async getConversation(conversationId: string): Promise<CONVERSATION> {
    return await this.sqlitePresenter.getConversation(conversationId)
  }

  async updateConversationTitle(conversationId: string, title: string): Promise<void> {
    await this.sqlitePresenter.updateConversation(conversationId, { title })
  }

  async updateConversationSettings(
    conversationId: string,
    settings: Partial<CONVERSATION_SETTINGS>
  ): Promise<void> {
    const conversation = await this.getConversation(conversationId)
    const mergedSettings = { ...conversation.settings, ...settings }
    console.log('updateConversationSettings', mergedSettings)
    // 检查是否有 modelId 的变化
    if (settings.modelId && settings.modelId !== conversation.settings.modelId) {
      // 获取模型配置
      const modelConfig = getModelConfig(mergedSettings.modelId)
      console.log('check model default config', modelConfig)
      if (modelConfig) {
        // 如果当前设置小于推荐值，则使用推荐值
        mergedSettings.maxTokens = modelConfig.maxTokens
        mergedSettings.contextLength = modelConfig.contextLength
      }
    }

    await this.sqlitePresenter.updateConversation(conversationId, { settings: mergedSettings })
  }

  async getConversationList(
    page: number,
    pageSize: number
  ): Promise<{ total: number; list: CONVERSATION[] }> {
    return await this.sqlitePresenter.getConversationList(page, pageSize)
  }

  async setActiveConversation(conversationId: string): Promise<void> {
    const conversation = await this.getConversation(conversationId)
    if (conversation) {
      this.activeConversationId = conversationId
      eventBus.emit(CONVERSATION_EVENTS.ACTIVATED, { conversationId })
    } else {
      throw new Error(`Conversation ${conversationId} not found`)
    }
  }

  async getActiveConversation(): Promise<CONVERSATION | null> {
    if (!this.activeConversationId) {
      return null
    }
    return this.getConversation(this.activeConversationId)
  }

  async getMessages(
    conversationId: string,
    page: number,
    pageSize: number
  ): Promise<{ total: number; list: Message[] }> {
    return await this.messageManager.getMessageThread(conversationId, page, pageSize)
  }

  async getContextMessages(conversationId: string): Promise<Message[]> {
    const conversation = await this.getConversation(conversationId)
    // 计算需要获取的消息数量（假设每条消息平均300字）
    let messageCount = Math.ceil(conversation.settings.contextLength / 300)
    if (messageCount < 2) {
      messageCount = 2
    }
    return await this.messageManager.getContextMessages(conversationId, messageCount)
  }

  async clearContext(conversationId: string): Promise<void> {
    await this.sqlitePresenter.runTransaction(async () => {
      const conversation = await this.getConversation(conversationId)
      if (conversation) {
        await this.sqlitePresenter.deleteAllMessages()
      }
    })
  }
  /**
   *
   * @param conversationId
   * @param content
   * @param role
   * @returns 如果是user的消息，返回ai生成的message，否则返回空
   */
  async sendMessage(
    conversationId: string,
    content: string,
    role: MESSAGE_ROLE
  ): Promise<AssistantMessage | null> {
    const conversation = await this.getConversation(conversationId)
    const { providerId, modelId } = conversation.settings
    console.log('sendMessage', conversation)
    const message = await this.messageManager.sendMessage(
      conversationId,
      content,
      role,
      '',
      false,
      {
        totalTokens: 0,
        generationTime: 0,
        firstTokenTime: 0,
        tokensPerSecond: 0,
        inputTokens: 0,
        outputTokens: 0,
        model: modelId,
        provider: providerId
      }
    )
    if (role === 'user') {
      const assistantMessage = await this.generateAIResponse(conversationId, message.id)
      this.generatingMessages.set(assistantMessage.id, {
        message: assistantMessage,
        conversationId,
        startTime: Date.now(),
        firstTokenTime: null,
        promptTokens: 0,
        reasoningStartTime: null,
        reasoningEndTime: null,
        lastReasoningTime: null
      })

      // 检查是否是新会话的第一条消息
      const { list: messages } = await this.getMessages(conversationId, 1, 2)
      if (messages.length === 1) {
        // 更新会话的 is_new 标志位
        await this.sqlitePresenter.updateConversation(conversationId, { is_new: 0 })
      }

      return assistantMessage
    }

    return null
  }

  private async generateAIResponse(conversationId: string, userMessageId: string) {
    try {
      const triggerMessage = await this.messageManager.getMessage(userMessageId)
      if (!triggerMessage) {
        throw new Error('找不到触发消息')
      }

      await this.messageManager.updateMessageStatus(userMessageId, 'sent')

      const conversation = await this.getConversation(conversationId)
      const { providerId, modelId } = conversation.settings
      const assistantMessage = (await this.messageManager.sendMessage(
        conversationId,
        JSON.stringify([]),
        'assistant',
        userMessageId,
        false,
        {
          totalTokens: 0,
          generationTime: 0,
          firstTokenTime: 0,
          tokensPerSecond: 0,
          inputTokens: 0,
          outputTokens: 0,
          model: modelId,
          provider: providerId
        }
      )) as AssistantMessage

      this.generatingMessages.set(assistantMessage.id, {
        message: assistantMessage,
        conversationId,
        startTime: Date.now(),
        firstTokenTime: null,
        promptTokens: 0,
        reasoningStartTime: null,
        reasoningEndTime: null,
        lastReasoningTime: null
      })

      return assistantMessage
    } catch (error) {
      await this.messageManager.updateMessageStatus(userMessageId, 'error')
      console.error('生成 AI 响应失败:', error)
      throw error
    }
  }

  async getMessage(messageId: string): Promise<Message> {
    return await this.messageManager.getMessage(messageId)
  }

  /**
   * 获取指定消息之前的历史消息
   * @param messageId 消息ID
   * @param limit 限制返回的消息数量
   * @returns 历史消息列表，按时间正序排列
   */
  private async getMessageHistory(messageId: string, limit: number = 100): Promise<Message[]> {
    const message = await this.messageManager.getMessage(messageId)
    if (!message) {
      throw new Error('找不到指定的消息')
    }

    const { list: messages } = await this.messageManager.getMessageThread(
      message.conversationId,
      1,
      limit * 2
    )

    // 找到目标消息在列表中的位置
    const targetIndex = messages.findIndex((msg) => msg.id === messageId)
    if (targetIndex === -1) {
      return [message]
    }

    // 返回目标消息之前的消息（包括目标消息）
    return messages.slice(Math.max(0, targetIndex - limit + 1), targetIndex + 1)
  }

  private async rewriteUserSearchQuery(
    query: string,
    contextMessages: string,
    conversationId: string,
    searchEngine: string
  ): Promise<string> {
    const rewritePrompt = `
    你是一个搜索优化专家。基于以下内容，生成一个优化的搜索查询：

    当前时间：${new Date().toISOString()}
    搜索引擎：${searchEngine}

    请遵循以下规则重写搜索查询：
    1. 根据用户的问题和上下文，重写应该进行搜索的关键词
    2. 如果需要使用时间，则根据当前时间给出需要查询的具体时间日期信息
    3. 编程相关查询：
        - 加上编程语言或框架名称
        - 指定错误代码或具体版本号
    4. 保持查询简洁，通常不超过5-6个关键词
    5. 默认保留用户的问题的语言，如果用户的问题是中文，则返回中文，如果用户的问题是英文，则返回英文，其他语言也一样
    6. 如果用户的内容非常简单的字符或者词汇，没有特别的含义，直接返回原文，忽略以上1-5的规则

    直接返回优化后的搜索词，不要有任何额外说明。
    如下是之前对话的上下文：
    <context_messages>
    ${contextMessages}
    </context_messages>
    如下是用户的问题：
    <user_question>
    ${query}
    </user_question>
    `
    const conversation = await this.getConversation(conversationId)
    if (!conversation) {
      return query
    }
    // console.log('rewriteUserSearchQuery', query, contextMessages, conversation.id)
    const { providerId, modelId } = conversation.settings
    try {
      const rewrittenQuery = await this.llmProviderPresenter.generateCompletion(
        this.searchAssistantProviderId || providerId,
        [
          {
            role: 'user',
            content: rewritePrompt
          }
        ],
        this.searchAssistantModel?.id || modelId
      )
      console.log('rewriteUserSearchQuery', rewrittenQuery)
      return rewrittenQuery.trim() || query
    } catch (error) {
      console.error('重写搜索查询失败:', error)
      return query
    }
  }

  private async startStreamSearch(
    conversationId: string,
    messageId: string,
    query: string
  ): Promise<SearchResult[]> {
    const state = this.generatingMessages.get(messageId)
    if (!state) {
      throw new Error('找不到生成状态')
    }

    // 添加搜索加载状态
    const searchBlock: AssistantMessageBlock = {
      type: 'search',
      content: '',
      status: 'loading',
      timestamp: Date.now(),
      extra: {
        total: 0
      }
    }
    state.message.content.unshift(searchBlock)
    await this.messageManager.editMessage(messageId, JSON.stringify(state.message.content))

    try {
      // 获取历史消息用于上下文
      const contextMessages = await this.getContextMessages(conversationId)
      const formattedContext = contextMessages
        .map((msg) => {
          if (msg.role === 'user') {
            return `user: ${msg.content.text}${getFileContext(msg.content.files)}`
          } else if (msg.role === 'ai') {
            return `assistant: ${msg.content.blocks.map((block) => block.content).join('')}`
          } else {
            return JSON.stringify(msg.content)
          }
        })
        .join('\n')

      // 重写搜索查询
      const optimizedQuery = await this.rewriteUserSearchQuery(
        query,
        formattedContext,
        conversationId,
        this.searchManager.getActiveEngine().name
      )

      // 开始搜索
      const results = await this.searchManager.search(conversationId, optimizedQuery)

      // 更新搜索状态为阅读中
      searchBlock.status = 'reading'
      searchBlock.extra = {
        total: results.length
      }
      await this.messageManager.editMessage(messageId, JSON.stringify(state.message.content))

      // 保存搜索结果
      for (const result of results) {
        // console.log('保存搜索结果', result)
        await this.sqlitePresenter.addMessageAttachment(
          messageId,
          'search_result',
          JSON.stringify({
            title: result.title,
            url: result.url,
            content: result.content || '',
            description: result.description || '',
            icon: result.icon || ''
          })
        )
      }

      // 更新搜索状态为成功
      searchBlock.status = 'success'
      await this.messageManager.editMessage(messageId, JSON.stringify(state.message.content))

      return results
    } catch (error) {
      // 更新搜索状态为错误
      searchBlock.status = 'error'
      searchBlock.content = String(error)
      await this.messageManager.editMessage(messageId, JSON.stringify(state.message.content))
      return []
    }
  }

  private async getLastUserMessage(conversationId: string): Promise<Message | null> {
    return await this.messageManager.getLastUserMessage(conversationId)
  }

  // 从数据库获取搜索结果
  async getSearchResults(messageId: string): Promise<SearchResult[]> {
    const results = await this.sqlitePresenter.getMessageAttachments(messageId, 'search_result')
    return results.map((result) => JSON.parse(result.content) as SearchResult) ?? []
  }

  async startStreamCompletion(conversationId: string, queryMsgId?: string) {
    const state = Array.from(this.generatingMessages.values()).find(
      (state) => state.conversationId === conversationId
    )
    if (!state) {
      console.warn('未找到状态，conversationId:', conversationId)
      return
    }

    const conversation = await this.getConversation(conversationId)
    const { systemPrompt, providerId, modelId, temperature, contextLength, maxTokens, artifacts } =
      conversation.settings

    let contextMessages: Message[] = []
    let userMessage: Message | null = null
    let searchResults: SearchResult[] | null = null
    let urlResults: SearchResult[] = []

    try {
      if (queryMsgId) {
        console.log('有queryMsgId，从该消息开始获取历史消息')
        const queryMessage = await this.getMessage(queryMsgId)
        if (!queryMessage || !queryMessage.parentId) {
          console.error('找不到指定的消息，queryMsgId:', queryMsgId)
          throw new Error('找不到指定的消息')
        }
        userMessage = await this.getMessage(queryMessage.parentId)
        if (!userMessage) {
          console.error('找不到触发消息，parentId:', queryMessage.parentId)
          throw new Error('找不到指定的消息')
        }
        // 获取触发消息之前的历史消息
        contextMessages = await this.getMessageHistory(userMessage.id, contextLength)
      } else {
        // 直接从数据库获取最后一条用户消息
        userMessage = await this.getLastUserMessage(conversationId)
        if (!userMessage) {
          throw new Error('找不到用户消息')
        }
        contextMessages = await this.getContextMessages(conversationId)
      }

      if (!userMessage) {
        throw new Error('找不到用户消息')
      }

      function getAnswer(files) {
        const text = userMessage?.content.text || ''
        const file = files.find(item => item.content.indexOf(text) > -1)
        if (!file) {
          return files
        }

        const textIndex = file.content.indexOf(text)
        const questionStart = file.content.slice(0, textIndex).lastIndexOf('question')
        const questionEnd = textIndex + file.content.slice(textIndex).indexOf('question')

        const content = file.content.slice(questionStart, questionEnd) + '以 answer 的内容为准'

        return [{ ...file, content }]
      }

      // 处理本地文本信息
      const userContent = `
      ${userMessage.content.text}
      ${getFileContext(getAnswer(userMessage.content.files))}
      `
      // 从用户消息中提取并丰富URL内容
      urlResults = await ContentEnricher.extractAndEnrichUrls(userMessage.content.text)

      // 处理搜索
      if (userMessage.content.search) {
        searchResults = await this.startStreamSearch(conversationId, state.message.id, userContent)
      }

      // 计算搜索提示词的token数量
      const searchPrompt = searchResults ? generateSearchPrompt(userContent, searchResults) : ''

      // 使用URL内容丰富用户消息
      const enrichedUserMessage =
        urlResults.length > 0
          ? ContentEnricher.enrichUserMessageWithUrlContent(userContent, urlResults)
          : userContent

      // 计算token数量
      const searchPromptTokens = searchPrompt ? approximateTokenSize(searchPrompt) : 0
      const systemPromptTokens = systemPrompt ? approximateTokenSize(systemPrompt) : 0
      const userMessageTokens = approximateTokenSize(enrichedUserMessage)

      // 计算剩余可用的上下文长度
      const reservedTokens = searchPromptTokens + systemPromptTokens + userMessageTokens
      const remainingContextLength = contextLength - reservedTokens

      // 获取历史消息，优先考虑上下文边界
      if (remainingContextLength > 0) {
        const messages = contextMessages
          .filter((msg) => msg.id !== userMessage?.id) // 排除当前用户消息
          .reverse() // 从新到旧排序
        let currentLength = 0
        const selectedMessages: Message[] = []

        for (const msg of messages) {
          // 处理本地文本信息

          const msgTokens = approximateTokenSize(
            msg.role === 'user'
              ? `${msg.content.text}${getFileContext(msg.content.files)}`
              : JSON.stringify(msg.content)
          )

          if (currentLength + msgTokens <= remainingContextLength) {
            selectedMessages.unshift(msg)
            currentLength += msgTokens
          } else {
            break
          }
        }

        contextMessages = selectedMessages
      } else {
        contextMessages = []
      }

      const formattedMessages: { role: 'system' | 'user' | 'assistant'; content: string }[] = []

      // 添加系统提示语
      if (systemPrompt) {
        if (artifacts === 1) {
          const artifactsPrompt = await getArtifactsPrompt()
          formattedMessages.push({
            role: 'system',
            content: `${systemPrompt}\n\n${artifactsPrompt}`
          })
        } else {
          formattedMessages.push({
            role: 'system',
            content: systemPrompt
          })
        }
      } else {
        if (artifacts === 1) {
          const artifactsPrompt = await getArtifactsPrompt()
          formattedMessages.push({
            role: 'system',
            content: `${artifactsPrompt}`
          })
        }
      }
      console.log('contextMessages:', contextMessages)
      // 添加上下文消息
      contextMessages.forEach((msg) => {
        const content =
          msg.role === 'user'
            ? `${msg.content.text}${getFileContext(msg.content.files)}`
            : msg.content
                .filter((block) => block.type === 'content')
                .map((block) => block.content)
                .join('\n')

        if (msg.role === 'assistant' && !content) {
          return // 如果是assistant且content为空，则不加入formattedMessages
        }

        formattedMessages.push({
          role: msg.role as 'user' | 'assistant',
          content
        })
      })

      // 添加当前用户消息，如果有搜索结果则替换为搜索提示词
      formattedMessages.push({
        role: 'user',
        content: searchPrompt || enrichedUserMessage
      })

      const mergedMessages: { role: 'user' | 'assistant' | 'system'; content: string }[] = []
      for (let i = 0; i < formattedMessages.length; i++) {
        const currentMessage = formattedMessages[i]
        if (
          mergedMessages.length > 0 &&
          mergedMessages[mergedMessages.length - 1].role === currentMessage.role
        ) {
          mergedMessages[mergedMessages.length - 1].content += `\n${currentMessage.content}`
        } else {
          mergedMessages.push({ ...currentMessage })
        }
      }
      formattedMessages.length = 0 // 清空原数组
      formattedMessages.push(...mergedMessages) // 将合并后的消息推入原数组

      // 计算prompt tokens
      let promptTokens = 0
      for (const msg of formattedMessages) {
        promptTokens += approximateTokenSize(msg.content)
      }
      console.log('formattedMessage:', formattedMessages, 'promptTokens:', promptTokens)

      // 更新生成状态
      this.generatingMessages.set(state.message.id, {
        ...state,
        startTime: Date.now(),
        firstTokenTime: null,
        promptTokens
      })

      // 更新消息的usage信息
      await this.messageManager.updateMessageMetadata(state.message.id, {
        totalTokens: promptTokens,
        generationTime: 0,
        firstTokenTime: 0,
        tokensPerSecond: 0
      })

      await this.llmProviderPresenter.startStreamCompletion(
        providerId,
        formattedMessages,
        modelId,
        state.message.id,
        temperature,
        maxTokens
      )
    } catch (error) {
      console.error('流式生成过程中出错:', error)
      await this.handleMessageError(state.message.id, String(error))
      throw error
    }
  }

  async editMessage(messageId: string, content: string): Promise<Message> {
    return await this.messageManager.editMessage(messageId, content)
  }

  async deleteMessage(messageId: string): Promise<void> {
    await this.messageManager.deleteMessage(messageId)
  }

  async retryMessage(messageId: string): Promise<AssistantMessage> {
    const message = await this.messageManager.getMessage(messageId)
    if (message.role !== 'assistant') {
      throw new Error('只能重试助手消息')
    }

    const userMessage = await this.messageManager.getMessage(message.parentId || '')
    if (!userMessage) {
      throw new Error('找不到对应的用户消息')
    }
    const conversation = await this.getConversation(message.conversationId)
    const { providerId, modelId } = conversation.settings
    const assistantMessage = await this.messageManager.retryMessage(messageId, {
      totalTokens: 0,
      generationTime: 0,
      firstTokenTime: 0,
      tokensPerSecond: 0,
      inputTokens: 0,
      outputTokens: 0,
      model: modelId,
      provider: providerId
    })

    // 初始化生成状态
    this.generatingMessages.set(assistantMessage.id, {
      message: assistantMessage,
      conversationId: message.conversationId,
      startTime: Date.now(),
      firstTokenTime: null,
      promptTokens: 0,
      reasoningStartTime: null,
      reasoningEndTime: null,
      lastReasoningTime: null
    })

    return assistantMessage
  }

  async getMessageVariants(messageId: string): Promise<Message[]> {
    return await this.messageManager.getMessageVariants(messageId)
  }

  async updateMessageStatus(messageId: string, status: MESSAGE_STATUS): Promise<void> {
    await this.messageManager.updateMessageStatus(messageId, status)
  }

  async updateMessageMetadata(
    messageId: string,
    metadata: Partial<MESSAGE_METADATA>
  ): Promise<void> {
    await this.messageManager.updateMessageMetadata(messageId, metadata)
  }

  async markMessageAsContextEdge(messageId: string, isEdge: boolean): Promise<void> {
    await this.messageManager.markMessageAsContextEdge(messageId, isEdge)
  }

  async getActiveConversationId(): Promise<string | null> {
    return this.activeConversationId
  }

  private async getLatestConversation(): Promise<CONVERSATION | null> {
    const result = await this.getConversationList(1, 1)
    return result.list[0] || null
  }

  getGeneratingMessageState(messageId: string): GeneratingMessageState | null {
    return this.generatingMessages.get(messageId) || null
  }

  getConversationGeneratingMessages(conversationId: string): AssistantMessage[] {
    return Array.from(this.generatingMessages.values())
      .filter((state) => state.conversationId === conversationId)
      .map((state) => state.message)
  }

  async stopMessageGeneration(messageId: string): Promise<void> {
    const state = this.generatingMessages.get(messageId)
    if (state) {
      // 添加用户取消的消息块
      state.message.content.forEach((block) => {
        if (block.status === 'loading') {
          block.status = 'success'
        }
      })
      state.message.content.push({
        type: 'error',
        content: 'common.error.userCanceledGeneration',
        status: 'cancel',
        timestamp: Date.now()
      })

      // 更新消息状态和内容
      await this.messageManager.updateMessageStatus(messageId, 'error')
      await this.messageManager.editMessage(messageId, JSON.stringify(state.message.content))

      // 停止流式生成
      await this.llmProviderPresenter.stopStream(messageId)

      // 清理生成状态
      this.generatingMessages.delete(messageId)
    }
  }

  async stopConversationGeneration(conversationId: string): Promise<void> {
    const messageIds = Array.from(this.generatingMessages.entries())
      .filter(([, state]) => state.conversationId === conversationId)
      .map(([messageId]) => messageId)

    await Promise.all(messageIds.map((messageId) => this.stopMessageGeneration(messageId)))
  }

  async summaryTitles(providerId?: string, modelId?: string): Promise<string> {
    const conversation = await this.getActiveConversation()
    if (!conversation) {
      throw new Error('找不到当前对话')
    }
    if (!modelId) {
      modelId = conversation.settings.modelId
    }
    let summaryProviderId = providerId
    if (!summaryProviderId) {
      summaryProviderId = conversation.settings.providerId
    }
    const messages = await this.getContextMessages(conversation.id)
    const messagesWithLength = messages
      .map((msg) => {
        if (msg.role === 'user') {
          return {
            message: msg,
            length: `${msg.content.text}${getFileContext(msg.content.files)}`.length,
            formattedMessage: {
              role: 'user' as const,
              content: `${msg.content.text}${getFileContext(msg.content.files)}`
            }
          }
        } else {
          const content = msg.content
            .filter((block) => block.type === 'content')
            .map((block) => block.content)
            .join('\n')
          return {
            message: msg,
            length: content.length,
            formattedMessage: {
              role: 'assistant' as const,
              content: content
            }
          }
        }
      })
      .filter((item) => item.formattedMessage.content.length > 0)
    return await this.llmProviderPresenter.summaryTitles(
      messagesWithLength.map((item) => item.formattedMessage),
      summaryProviderId,
      modelId
    )
  }
  async clearActiveThread(): Promise<void> {
    this.activeConversationId = null
    eventBus.emit(CONVERSATION_EVENTS.DEACTIVATED)
  }

  async clearAllMessages(conversationId: string): Promise<void> {
    await this.messageManager.clearAllMessages(conversationId)
    // 如果是当前活动会话，需要更新生成状态
    if (conversationId === this.activeConversationId) {
      // 停止所有正在生成的消息
      await this.stopConversationGeneration(conversationId)
    }
  }

  async getMessageExtraInfo(messageId: string, type: string): Promise<Record<string, unknown>[]> {
    const attachments = await this.sqlitePresenter.getMessageAttachments(messageId, type)
    return attachments.map((attachment) => JSON.parse(attachment.content))
  }

  async getMainMessageByParentId(
    conversationId: string,
    parentId: string
  ): Promise<Message | null> {
    const message = await this.messageManager.getMainMessageByParentId(conversationId, parentId)
    if (!message) {
      return null
    }
    return message
  }

  destroy() {
    this.searchManager.destroy()
  }
}
