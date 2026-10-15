import { Button, Collapse, Image, Tag, Typography } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { AssistantMsg, Msg, Reference } from '../types'

const STEP_LABELS: Record<string, string> = {
  dispatch: '调度',
  staff: '员工',
  tool: '工具',
  result: '结果',
  info: '信息',
}

// 引用来源展示的收敛规则：Agent 在一轮对话里可能反复调用知识检索，
// 每次都推一批来源，前端若不去重会把几十个（甚至上百个）标签全堆出来。
// 这里按「去重 + 相关度阈值 + 条数上限」收敛成少量有代表性的来源。
const MIN_REF_SCORE = 0.1
const MAX_REF_CHIPS = 8

function toScore(v: number | string | undefined): number {
  if (v == null) return 0
  const n = typeof v === 'number' ? v : Number.parseFloat(v)
  return Number.isFinite(n) ? n : 0
}

function dedupeRefs(refs: Reference[]): Reference[] {
  // 同一来源只保留相关度最高的一条，再过滤低相关度、按相关度降序
  const best = new Map<string, Reference>()
  for (const r of refs) {
    const prev = best.get(r.source)
    if (!prev || toScore(r.score) > toScore(prev.score)) best.set(r.source, r)
  }
  return Array.from(best.values())
    .filter((r) => toScore(r.score) >= MIN_REF_SCORE)
    .sort((a, b) => toScore(b.score) - toScore(a.score))
}

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
      }}
    >
      {text}
    </ReactMarkdown>
  )
}

export default function MessageBubble({ msg }: { msg: Msg }) {
  if (msg.kind === 'user') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 16 }}>
        <div style={{ maxWidth: '76%', background: '#1677ff', color: '#fff', padding: '9px 14px', borderRadius: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
          {msg.text}
        </div>
      </div>
    )
  }

  const a = msg as AssistantMsg
  const distinctRefs = dedupeRefs(a.refs)
  const visibleRefs = distinctRefs.slice(0, MAX_REF_CHIPS)
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <div style={{ maxWidth: '96%', width: '100%', background: '#fff', border: '1px solid #eee', borderRadius: 12, padding: '10px 14px' }}>
          {a.steps.length > 0 && (
            <Collapse
              ghost
              size="small"
              style={{ marginBottom: 8 }}
              items={[{
                key: 'trace',
                label: <Typography.Text type="secondary" style={{ fontSize: 12 }}>执行过程（{a.steps.length} 步）</Typography.Text>,
                children: (
                  <div>
                    {a.steps.map((s) => (
                      <div key={s.id} style={{ fontSize: 12, marginBottom: 6, lineHeight: 1.6 }}>
                        <Tag style={{ marginRight: 6, fontSize: 11 }}>{STEP_LABELS[s.kind] ?? s.kind}</Tag>
                        <span style={{ color: '#555' }}>{s.title}</span>
                        {s.detail && (
                          <pre style={{ margin: '4px 0 0', padding: 6, background: '#f7f7f7', borderRadius: 6, whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: 11, color: '#888' }}>{s.detail}</pre>
                        )}
                      </div>
                    ))}
                  </div>
                ),
              }]}
            />
          )}
          <div style={{ lineHeight: 1.7 }}>
            {a.done ? (
              a.text ? <Markdown text={a.text} /> : <Typography.Text type="secondary">（无内容）</Typography.Text>
            ) : (
              <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{a.text || '思考中…'}</div>
            )}
          </div>
          {distinctRefs.length > 0 && (
            <div style={{ marginTop: 8 }}>
              {visibleRefs.map((r) => (
                <Tag key={r.source} color="blue" style={{ marginBottom: 4 }}>📄 {r.source}{r.score != null ? `（${r.score}）` : ''}</Tag>
              ))}
              {distinctRefs.length > MAX_REF_CHIPS && (
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>… 另有 {distinctRefs.length - MAX_REF_CHIPS} 个来源未显示</Typography.Text>
              )}
            </div>
          )}
          {a.media.map((m) =>
            m.kind === 'image' ? (
              <div key={m.id} style={{ marginTop: 8 }}>
                <Image src={m.url} alt={m.title || '趋势图'} style={{ maxWidth: '100%' }} />
              </div>
            ) : (
              <div key={m.id} style={{ marginTop: 8 }}>
                <Button type="link" icon={<FileTextOutlined />} href={m.url} target="_blank" style={{ paddingLeft: 0 }}>
                  下载报告：{m.filename || m.title || m.url}
                </Button>
              </div>
            )
          )}
          {a.error && <Typography.Text type="danger" style={{ display: 'block', marginTop: 6 }}>{a.error}</Typography.Text>}
        </div>
      </div>
    </div>
  )
}
