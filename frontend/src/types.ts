export type Role = 'ops' | 'leader' | 'service'

export interface UserInfo {
  id: number
  username: string
  name: string
  role: Role
  companies?: { id: number; name: string }[]
}

export interface Proposal {
  device_id?: number | string
  param?: string
  param_label?: string
  old_value?: string | null
  new_value?: string
}

export interface ConfirmPayload {
  title?: string
  question?: string
  proposal?: Proposal
}

export interface Step {
  id: string
  kind: 'dispatch' | 'staff' | 'tool' | 'result' | 'info'
  title: string
  detail?: string
}

export interface Reference {
  source: string
  score?: number | string
}

export interface MediaItem {
  id: string
  kind: 'image' | 'file'
  url: string
  title?: string
  filename?: string
}

export interface UserMsg {
  id: string
  kind: 'user'
  text: string
}

export interface AssistantMsg {
  id: string
  kind: 'assistant'
  text: string
  done: boolean
  error?: string
  steps: Step[]
  refs: Reference[]
  media: MediaItem[]
}

export type Msg = UserMsg | AssistantMsg

export type ChatEvent =
  | { type: 'content.delta'; delta: string }
  | { type: 'dispatch'; staff: string; args?: unknown }
  | { type: 'staff.start'; staff: string; task?: string }
  | { type: 'tool.call'; staff?: string; name: string; args?: unknown }
  | { type: 'tool.result'; staff?: string; name?: string; output?: string }
  | { type: 'staff.end'; staff: string; output?: string }
  | { type: 'reference.source'; source: string; score?: number }
  | { type: 'image'; url: string; title?: string }
  | { type: 'file'; url: string; filename?: string; title?: string }
  | { type: 'interrupt'; kind?: string; title?: string; question?: string; proposal?: Proposal }
  | { type: 'confirm.proposal'; proposal?: Proposal }
  | { type: 'write.applied'; param?: string; new_value?: string; device_id?: number | string }
  | { type: 'error'; message: string }
  | { type: 'complete'; answer: string }
