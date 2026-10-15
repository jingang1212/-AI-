import { useCallback, useEffect, useRef, useState } from 'react'
import { App as AntdApp, Button, Empty, Form, Input, Layout, Segmented, Tag, Typography } from 'antd'
import { LogoutOutlined, PlusOutlined, SendOutlined } from '@ant-design/icons'
import { confirmDecision, login } from './api'
import type { AssistantMsg, ChatEvent, ConfirmPayload, Msg, Role, UserInfo, UserMsg } from './types'
import ConfirmModal from './components/ConfirmModal'
import MessageBubble from './components/MessageBubble'

const ROLE_LABELS: Record<Role, string> = { leader: '环保局领导', ops: '安装运维', service: '客服' }
const ROLE_COLORS: Record<Role, string> = { leader: 'gold', ops: 'blue', service: 'green' }

let seq = 0
function uid(prefix: string): string {
  seq += 1
  return `${prefix}-${Date.now()}-${seq}`
}

function emptyAssistant(id: string): AssistantMsg {
  return { id, kind: 'assistant', text: '', done: false, steps: [], refs: [], media: [] }
}

export default function App() {
  const { message } = AntdApp.useApp()
  const [token, setToken] = useState<string>(() => localStorage.getItem('ai-helper-token') || '')
  const [user, setUser] = useState<UserInfo | null>(() => {
    const raw = localStorage.getItem('ai-helper-user')
    if (!raw) return null
    try { return JSON.parse(raw) as UserInfo } catch { return null }
  })
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [protocol, setProtocol] = useState<'sse' | 'ws'>('sse')
  const [sessionId, setSessionId] = useState<string>(() => `s-${Date.now()}`)
  const [confirmPayload, setConfirmPayload] = useState<ConfirmPayload | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [loginError, setLoginError] = useState('')

  const currentIdRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const listRef = useRef<HTMLDivElement | null>(null)

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      const el = listRef.current
      if (el) el.scrollTop = el.scrollHeight
    })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  const patchAssistant = useCallback((id: string, patch: (a: AssistantMsg) => AssistantMsg) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? patch(m as AssistantMsg) : m)))
  }, [])

  const handleEvent = useCallback((evt: ChatEvent) => {
    const id = currentIdRef.current
    if (!id) return
    const addStep = (kind: AssistantMsg['steps'][number]['kind'], title: string, detail?: string) =>
      patchAssistant(id, (a) => ({ ...a, steps: [...a.steps, { id: uid('step'), kind, title, detail }] }))

    switch (evt.type) {
      case 'content.delta':
        patchAssistant(id, (a) => ({ ...a, text: a.text + evt.delta }))
        break
      case 'dispatch':
        addStep('dispatch', `主管调度：${evt.staff}`)
        break
      case 'staff.start':
        addStep('staff', `${evt.staff} 开始处理`, evt.task)
        break
      case 'tool.call':
        addStep('tool', `${evt.staff ? `${evt.staff} 调用 ` : ''}${evt.name}`, evt.args ? JSON.stringify(evt.args, null, 2) : undefined)
        break
      case 'tool.result':
        addStep('result', `${evt.name ?? '工具'} 返回`, evt.output)
        break
      case 'staff.end':
        addStep('info', `${evt.staff} 完成`, evt.output)
        break
      case 'reference.source':
        patchAssistant(id, (a) => ({ ...a, refs: [...a.refs, { source: evt.source, score: evt.score }] }))
        break
      case 'image':
        patchAssistant(id, (a) => ({ ...a, media: [...a.media, { id: uid('media'), kind: 'image', url: evt.url, title: evt.title }] }))
        break
      case 'file':
        patchAssistant(id, (a) => ({ ...a, media: [...a.media, { id: uid('media'), kind: 'file', url: evt.url, filename: evt.filename, title: evt.title }] }))
        break
      case 'confirm.proposal':
        addStep('info', '已生成修改提案，等待确认')
        break
      case 'interrupt':
        setConfirmPayload({ title: evt.title, question: evt.question, proposal: evt.proposal })
        break
      case 'write.applied':
        addStep('info', `修改已生效：${evt.param ?? ''} → ${evt.new_value ?? ''}`)
        break
      case 'error':
        patchAssistant(id, (a) => ({ ...a, error: evt.message, done: true }))
        setBusy(false)
        break
      case 'complete':
        patchAssistant(id, (a) => ({ ...a, text: evt.answer || a.text, done: true }))
        setBusy(false)
        break
    }
  }, [patchAssistant])

  const onLogin = async (values: { username: string; password: string }) => {
    setLoginError('')
    try {
      const { token: t, user: u } = await login(values.username, values.password)
      localStorage.setItem('ai-helper-token', t)
      localStorage.setItem('ai-helper-user', JSON.stringify(u))
      setToken(t)
      setUser(u)
      setMessages([])
      setSessionId(`s-${Date.now()}`)
      message.success(`欢迎，${u.name}`)
    } catch (e) {
      setLoginError((e as Error).message)
    }
  }

  const logout = () => {
    abortRef.current?.abort()
    wsRef.current?.close()
    localStorage.removeItem('ai-helper-token')
    localStorage.removeItem('ai-helper-user')
    setToken('')
    setUser(null)
    setMessages([])
    setConfirmPayload(null)
    setBusy(false)
  }

  const newSession = () => {
    if (busy) { message.warning('当前会话进行中，请先停止或等它结束'); return }
    setMessages([])
    setSessionId(`s-${Date.now()}`)
  }

  const stop = useCallback(() => {
    if (protocol === 'sse') abortRef.current?.abort()
    else wsRef.current?.send(JSON.stringify({ type: 'abandon' }))
  }, [protocol])

  const send = useCallback(async () => {
    const text = input.trim()
    if (!text || busy || !token) return
    setInput('')
    setBusy(true)
    const userMsg: UserMsg = { id: uid('u'), kind: 'user', text }
    setMessages((prev) => [...prev, userMsg])
    const assistantId = uid('a')
    currentIdRef.current = assistantId
    setMessages((prev) => [...prev, emptyAssistant(assistantId)])
    try {
      if (protocol === 'sse') {
        const controller = new AbortController()
        abortRef.current = controller
        const resp = await fetch('/sse/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ message: text, session_id: sessionId }),
          signal: controller.signal,
        })
        if (!resp.ok || !resp.body) throw new Error(`请求失败（HTTP ${resp.status}）`)
        const reader = resp.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const parts = buffer.split('\n\n')
          buffer = parts.pop() ?? ''
          for (const part of parts) {
            const line = part.trim()
            if (!line.startsWith('data:')) continue
            const payload = line.slice(5).trim()
            if (!payload) continue
            try { handleEvent(JSON.parse(payload) as ChatEvent) } catch { /* 忽略损坏事件 */ }
          }
        }
      } else {
        await new Promise<void>((resolve) => {
          const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
          const ws = new WebSocket(`${scheme}://${location.host}/ws/chat?token=${encodeURIComponent(token)}`)
          wsRef.current = ws
          ws.onopen = () => ws.send(JSON.stringify({ type: 'chat', message: text, session_id: sessionId }))
          ws.onmessage = (e) => { try { handleEvent(JSON.parse(e.data) as ChatEvent) } catch { /* 忽略 */ } }
          ws.onerror = () => {
            patchAssistant(assistantId, (a) => ({ ...a, error: 'WebSocket 连接失败', done: true }))
            resolve()
          }
          ws.onclose = () => resolve()
        })
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        patchAssistant(assistantId, (a) => ({ ...a, error: (e as Error).message || '请求异常', done: true }))
      }
    } finally {
      setBusy(false)
      currentIdRef.current = null
    }
  }, [input, busy, token, protocol, sessionId, handleEvent, patchAssistant])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && busy) stop()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, stop])

  const onApprove = async () => {
    setSubmitting(true)
    try {
      await confirmDecision(sessionId, true)
      message.success('已批准，AI 继续执行…')
      setConfirmPayload(null)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const onReject = async () => {
    setSubmitting(true)
    try {
      await confirmDecision(sessionId, false)
      message.info('已拒绝本次修改')
      setConfirmPayload(null)
    } catch (e) {
      message.error((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  if (!token || !user) {
    return (
      <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ width: 360, background: '#fff', padding: 28, borderRadius: 12, boxShadow: '0 2px 12px rgba(0,0,0,0.08)' }}>
          <Typography.Title level={4} style={{ textAlign: 'center', marginBottom: 4 }}>环保监测智能运维 AI 助手</Typography.Title>
          <Typography.Paragraph type="secondary" style={{ textAlign: 'center' }}>请登录后使用</Typography.Paragraph>
          <Form onFinish={onLogin} layout="vertical" requiredMark={false}>
            <Form.Item name="username" label="账号" rules={[{ required: true, message: '请输入账号' }]}>
              <Input placeholder="leader_01 / ops_01 / service_01" autoFocus />
            </Form.Item>
            <Form.Item name="password" label="密码" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password placeholder="123456" />
            </Form.Item>
            {loginError && <Typography.Text type="danger">{loginError}</Typography.Text>}
            <Button type="primary" htmlType="submit" block style={{ marginTop: 8 }}>登录</Button>
          </Form>
          <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0, fontSize: 12 }}>
            演示账号：leader_01（领导）/ ops_01（安装运维）/ service_01（客服），密码均为 123456
          </Typography.Paragraph>
        </div>
      </div>
    )
  }

  return (
    <Layout style={{ height: '100%' }}>
      <Layout.Header style={{ display: 'flex', alignItems: 'center', gap: 12, paddingInline: 16, background: '#fff', borderBottom: '1px solid #eee' }}>
        <Typography.Title level={5} style={{ margin: 0, whiteSpace: 'nowrap' }}>环保监测智能运维 AI 助手</Typography.Title>
        <Tag color={ROLE_COLORS[user.role]}>{ROLE_LABELS[user.role] ?? user.role}</Tag>
        <Typography.Text type="secondary">{user.name}</Typography.Text>
        <div style={{ flex: 1 }} />
        <Segmented
          value={protocol}
          onChange={(v) => setProtocol(v as 'sse' | 'ws')}
          options={[{ label: 'SSE', value: 'sse' }, { label: 'WebSocket', value: 'ws' }]}
        />
        <Button icon={<PlusOutlined />} onClick={newSession}>新会话</Button>
        <Button icon={<LogoutOutlined />} onClick={logout}>退出</Button>
      </Layout.Header>
      <Layout.Content style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div ref={listRef} style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
          {messages.length === 0 && (
            <Empty
              style={{ marginTop: 80 }}
              description="向 AI 助手提问，例如：3 号排口最近 24 小时二氧化硫趋势怎么样"
            />
          )}
          {messages.map((m) => <MessageBubble key={m.id} msg={m} />)}
        </div>
        <div style={{ padding: 12, background: '#fff', borderTop: '1px solid #eee' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
            <Input.TextArea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPressEnter={(e) => { if (!e.shiftKey) { e.preventDefault(); send() } }}
              placeholder={busy ? '回答中…（按 Esc 停止）' : '输入问题，Enter 发送，Shift+Enter 换行'}
              autoSize={{ minRows: 1, maxRows: 5 }}
              disabled={busy}
              style={{ flex: 1 }}
            />
            {busy
              ? <Button danger onClick={stop} style={{ height: 32 }}>停止</Button>
              : <Button type="primary" icon={<SendOutlined />} onClick={send} disabled={!input.trim()} style={{ height: 32 }}>发送</Button>}
          </div>
        </div>
      </Layout.Content>
      <ConfirmModal
        open={!!confirmPayload}
        payload={confirmPayload}
        submitting={submitting}
        onApprove={onApprove}
        onReject={onReject}
      />
    </Layout>
  )
}
