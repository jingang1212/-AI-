import type { UserInfo } from './types'

export async function login(username: string, password: string): Promise<{ token: string; user: UserInfo }> {
  const resp = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!resp.ok) {
    const data = await resp.json().catch(() => ({}))
    throw new Error(data.detail || `登录失败（HTTP ${resp.status}）`)
  }
  return resp.json()
}

export async function confirmDecision(sessionId: string, approved: boolean, comment = ''): Promise<boolean> {
  const token = localStorage.getItem('ai-helper-token') || ''
  const resp = await fetch('/api/chat/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({ session_id: sessionId, approved, comment }),
  })
  if (!resp.ok) throw new Error('确认请求失败')
  const data = await resp.json()
  return !!data.ok
}
