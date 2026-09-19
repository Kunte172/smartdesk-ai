import { FormEvent, useMemo, useState } from "react";

type Prediction = {
  category: string;
  priority: string;
  category_confidence: number;
  priority_confidence: number;
  confidence: number;
  needs_review: boolean;
  model_version: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const examples = [
  "Tôi không thể đăng nhập vào tài khoản. Việc này đang chặn công việc của cả nhóm, cần xử lý sớm.",
  "Khoản hoàn tiền chưa xuất hiện trong tài khoản. Mong nhận được phản hồi trong hôm nay.",
  "Ứng dụng bị treo khi mở trang chính. KHẨN CẤP: toàn bộ dịch vụ đã ngừng hoạt động.",
];

const categoryLabels: Record<string, string> = {
  ACCOUNT_ACCESS: "Truy cập tài khoản",
  BILLING: "Thanh toán",
  DELIVERY: "Giao hàng",
  GENERAL: "Thông tin chung",
  REFUND: "Hoàn tiền",
  TECHNICAL: "Kỹ thuật",
};

const priorityLabels: Record<string, string> = {
  LOW: "Thấp",
  MEDIUM: "Trung bình",
  HIGH: "Cao",
  URGENT: "Khẩn cấp",
};

function percent(value: number) {
  return new Intl.NumberFormat("vi-VN", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

export default function App() {
  const [text, setText] = useState(examples[0]);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const confidenceTone = useMemo(() => {
    if (!prediction) return "neutral";
    if (prediction.needs_review) return "warning";
    return "success";
  }, [prediction]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/v1/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail ?? `API trả về lỗi ${response.status}`);
      }
      setPrediction((await response.json()) as Prediction);
    } catch (caught) {
      setPrediction(null);
      setError(caught instanceof Error ? caught.message : "Không thể kết nối API");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <section className="hero">
        <div className="brand-mark">SD</div>
        <div>
          <p className="eyebrow">AI-ASSISTED SUPPORT OPERATIONS</p>
          <h1>SmartDesk AI</h1>
          <p className="subtitle">
            Phân loại ticket, ước lượng mức ưu tiên và chuyển trường hợp không chắc chắn cho
            người kiểm tra.
          </p>
        </div>
      </section>

      <section className="workspace">
        <form className="panel composer" onSubmit={submit}>
          <div className="panel-heading">
            <div>
              <p className="step">01 · INPUT</p>
              <h2>Nội dung yêu cầu hỗ trợ</h2>
            </div>
            <span className="counter">{text.length}/5000</span>
          </div>

          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            minLength={8}
            maxLength={5000}
            required
            aria-label="Nội dung ticket"
          />

          <div className="examples">
            <span>Thử nhanh</span>
            {examples.map((example, index) => (
              <button type="button" key={index} onClick={() => setText(example)}>
                Mẫu {index + 1}
              </button>
            ))}
          </div>

          <button className="primary" type="submit" disabled={loading || text.trim().length < 8}>
            {loading ? "Đang phân tích…" : "Phân tích ticket"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>

        <section className="panel result" aria-live="polite">
          <div className="panel-heading">
            <div>
              <p className="step">02 · TRIAGE RESULT</p>
              <h2>Kết quả phân loại</h2>
            </div>
            {prediction && <span className="model-tag">{prediction.model_version}</span>}
          </div>

          {!prediction ? (
            <div className="empty-state">
              <div className="scan-line" />
              <p>Kết quả sẽ xuất hiện sau khi gửi ticket.</p>
            </div>
          ) : (
            <div className="prediction-grid">
              <article>
                <span>Nhóm xử lý</span>
                <strong>{categoryLabels[prediction.category] ?? prediction.category}</strong>
                <small>{percent(prediction.category_confidence)} tin cậy</small>
              </article>
              <article>
                <span>Mức ưu tiên</span>
                <strong className={`priority priority-${prediction.priority.toLowerCase()}`}>
                  {priorityLabels[prediction.priority] ?? prediction.priority}
                </strong>
                <small>{percent(prediction.priority_confidence)} tin cậy</small>
              </article>

              <div className={`review-banner ${confidenceTone}`}>
                <div>
                  <span>Quyết định định tuyến</span>
                  <strong>
                    {prediction.needs_review ? "Cần người kiểm tra" : "Có thể tự động định tuyến"}
                  </strong>
                </div>
                <b>{percent(prediction.confidence)}</b>
              </div>
            </div>
          )}
        </section>
      </section>

      <footer>
        <span>Baseline: TF-IDF + Logistic Regression</span>
        <span>Confidence-aware routing</span>
        <span>Human-in-the-loop</span>
      </footer>
    </main>
  );
}

