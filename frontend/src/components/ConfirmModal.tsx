import { Descriptions, Modal, Typography } from 'antd'
import type { ConfirmPayload } from '../types'

interface Props {
  open: boolean
  payload: ConfirmPayload | null
  submitting: boolean
  onApprove: () => void
  onReject: () => void
}

export default function ConfirmModal({ open, payload, submitting, onApprove, onReject }: Props) {
  const p = payload?.proposal
  return (
    <Modal
      open={open}
      title={payload?.title || 'AI 助手请求修改设备配置'}
      okText="批准"
      cancelText="拒绝"
      confirmLoading={submitting}
      onOk={onApprove}
      onCancel={onReject}
      maskClosable={false}
      keyboard={false}
      closable={false}
    >
      {payload?.question && <Typography.Paragraph>{payload.question}</Typography.Paragraph>}
      {p && (
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="设备 ID">{String(p.device_id ?? '—')}</Descriptions.Item>
          <Descriptions.Item label="参数">{p.param_label || p.param || '—'}</Descriptions.Item>
          <Descriptions.Item label="原值">{p.old_value ?? '（未配置）'}</Descriptions.Item>
          <Descriptions.Item label="新值">{String(p.new_value ?? '—')}</Descriptions.Item>
        </Descriptions>
      )}
      <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0, fontSize: 12 }}>
        批准后将写入数据库并留审计日志；拒绝则不会发生任何写入。
      </Typography.Paragraph>
    </Modal>
  )
}
