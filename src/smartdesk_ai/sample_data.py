from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from smartdesk_ai.domain import TicketCategory, TicketPriority

CATEGORY_TEMPLATES: dict[TicketCategory, tuple[str, ...]] = {
    TicketCategory.ACCOUNT_ACCESS: (
        "Tôi không thể đăng nhập vào tài khoản",
        "Tài khoản của tôi đang bị khóa",
        "Tôi quên mật khẩu và cần lấy lại quyền truy cập",
        "Mã xác thực đăng nhập không được gửi tới điện thoại",
        "Phiên đăng nhập liên tục bị hết hạn",
        "Tôi muốn thay đổi email đăng nhập",
    ),
    TicketCategory.BILLING: (
        "Tôi bị tính phí hai lần cho cùng một dịch vụ",
        "Số tiền trên hóa đơn tháng này không chính xác",
        "Thẻ thanh toán bị từ chối nhưng vẫn bị giữ tiền",
        "Tôi cần cập nhật phương thức thanh toán",
        "Gói dịch vụ đã hủy nhưng vẫn phát sinh phí",
        "Tôi cần bản sao chứng từ thanh toán",
    ),
    TicketCategory.DELIVERY: (
        "Đơn hàng chưa được giao theo lịch dự kiến",
        "Trạng thái vận chuyển không thay đổi nhiều ngày",
        "Bưu kiện được báo đã giao nhưng tôi chưa nhận được",
        "Tôi muốn thay đổi địa chỉ nhận hàng",
        "Sản phẩm bị hư hỏng trong quá trình vận chuyển",
        "Đơn hàng chỉ được giao một phần",
    ),
    TicketCategory.GENERAL: (
        "Tôi muốn biết giờ làm việc của bộ phận hỗ trợ",
        "Cho tôi hỏi dịch vụ hiện hỗ trợ những khu vực nào",
        "Tôi cần thông tin về các gói dịch vụ hiện có",
        "Làm thế nào để cập nhật thông tin liên hệ",
        "Tôi muốn đóng góp ý kiến về trải nghiệm sử dụng",
        "Tôi cần hướng dẫn bắt đầu sử dụng sản phẩm",
    ),
    TicketCategory.REFUND: (
        "Tôi muốn yêu cầu hoàn tiền cho đơn đã hủy",
        "Khoản hoàn tiền chưa xuất hiện trong tài khoản",
        "Tôi nhận sai sản phẩm và muốn được hoàn tiền",
        "Yêu cầu hoàn tiền của tôi đang chờ xử lý",
        "Tôi cần kiểm tra trạng thái khoản tiền được hoàn",
        "Sản phẩm lỗi và tôi muốn trả lại để nhận hoàn tiền",
    ),
    TicketCategory.TECHNICAL: (
        "Ứng dụng bị treo mỗi khi tôi mở trang chính",
        "Tôi không thể tải tệp lên hệ thống",
        "Trang web hiển thị lỗi máy chủ khi lưu dữ liệu",
        "Thông báo không xuất hiện dù đã bật trong cài đặt",
        "Ứng dụng chạy rất chậm sau bản cập nhật",
        "Tính năng tìm kiếm trả về màn hình trắng",
    ),
}

PRIORITY_PHRASES: dict[TicketPriority, tuple[str, ...]] = {
    TicketPriority.LOW: (
        "Không gấp, vui lòng phản hồi khi thuận tiện.",
        "Tôi chỉ cần biết thêm thông tin trong vài ngày tới.",
        "Vấn đề này chưa ảnh hưởng đến công việc hiện tại.",
    ),
    TicketPriority.MEDIUM: (
        "Mong nhận được phản hồi trong hôm nay.",
        "Vấn đề đang gây bất tiện cho một người dùng.",
        "Xin hỗ trợ sớm để tôi tiếp tục sử dụng dịch vụ.",
    ),
    TicketPriority.HIGH: (
        "Việc này đang chặn công việc của cả nhóm, cần xử lý sớm.",
        "Nhiều người dùng đang bị ảnh hưởng nghiêm trọng.",
        "Chúng tôi không thể hoàn thành tác vụ quan trọng.",
    ),
    TicketPriority.URGENT: (
        "KHẨN CẤP: toàn bộ dịch vụ đã ngừng hoạt động.",
        "Cần xử lý ngay lập tức vì hệ thống đang gián đoạn hoàn toàn.",
        "Sự cố nghiêm trọng đang ảnh hưởng tất cả người dùng.",
    ),
}

CONTEXTS = (
    "Tình trạng bắt đầu từ sáng nay.",
    "Tôi đã thử lại trên trình duyệt khác nhưng chưa được.",
    "Đây là lần đầu tôi gặp tình trạng này.",
    "Tôi đã làm theo hướng dẫn cơ bản nhưng vấn đề vẫn còn.",
    "Thông tin liên quan đã được kiểm tra hai lần.",
    "Vui lòng cho biết thêm bước khắc phục nếu cần.",
)


def generate_sample_dataset(
    output_path: str | Path, *, samples_per_combination: int = 12, seed: int = 42
) -> pd.DataFrame:
    if samples_per_combination < 4:
        raise ValueError("samples_per_combination must be at least 4")

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    sequence = 1
    for category, category_templates in CATEGORY_TEMPLATES.items():
        for priority, priority_phrases in PRIORITY_PHRASES.items():
            for index in range(samples_per_combination):
                category_text = category_templates[index % len(category_templates)]
                priority_text = priority_phrases[index % len(priority_phrases)]
                context = CONTEXTS[(index + sequence) % len(CONTEXTS)]
                parts = [category_text, context, priority_text]
                if index % 2:
                    parts[1], parts[2] = parts[2], parts[1]
                text = " ".join(parts)
                rows.append(
                    {
                        "ticket_id": f"SYN-{sequence:05d}",
                        "text": text,
                        "category": category.value,
                        "priority": priority.value,
                    }
                )
                sequence += 1

    rng.shuffle(rows)
    data = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return data
