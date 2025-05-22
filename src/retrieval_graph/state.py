# file: state.py
from typing import List, Optional, TypedDict, Any
from pydantic import BaseModel, Field
from langgraph.prebuilt.chat_agent_executor import AgentState as LangGraphReactAgentState

class SubQuestion(BaseModel): # Pydantic v2 BaseModel
    text: str = Field(..., description="Văn bản của phần câu hỏi con.")
    intent: str = Field(..., description="Ý định chính của phần câu hỏi con.")
    entities: List[str] = Field(..., description="Danh sách thực thể/từ khóa chính của phần câu hỏi con.")
    dependencies: List[int] = Field(
        default_factory=list,
        description="Chỉ số các SubQuestion khác mà phần này phụ thuộc vào."
    )

class QueryAnalysisOutput(BaseModel): # Pydantic v2 BaseModel
    original_query: str = Field(..., description="Câu hỏi gốc của người dùng.")
    user_roles: Optional[List[str]] = Field(..., description="Danh sách vai trò gốc của người dùng.")
    asker_role_context: str = Field(..., description="Vai trò suy luận của người hỏi (ví dụ: 'nhân viên', 'quản lý').")
    intent: str = Field(..., description="Ý định chính suy luận từ câu hỏi của người dùng.")
    entities: List[str] = Field(..., description="Danh sách các thực thể hoặc từ khóa chính được trích xuất.")
    sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="Danh sách các phần (sub-questions) đã tách và phân tích từ câu hỏi gốc."
    )
    plan_steps: List[str] = Field(
        default_factory=list,
        description="Danh sách các bước hành động cần thực hiện để trả lời từng phần câu hỏi."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Các câu hỏi phụ cần hỏi lại người dùng nếu có chỗ chưa rõ."
    )
    complexity_level: Optional[str] = Field(
        default=None,
        description="Mức độ phức tạp của câu hỏi ('low', 'medium', 'high')."
    )
    effective_search_query: List[str] = Field(
        default_factory=list,
        description="Danh sách các truy vấn tối ưu cho việc tìm kiếm tài liệu nội bộ, tương ứng với từng sub-question."
    )
    status: str = Field(
        default="processed_for_main_agent",
        description="Trạng thái xử lý, luôn là 'processed_for_main_agent'."
    )

class AmelaReactCompatibleAgentState(LangGraphReactAgentState): # Kế thừa
    # messages, remaining_steps, is_last_step đã có từ base class
    # Input ban đầu
    original_query: str # Vẫn giữ để tiện truy cập query gốc của lượt hiện tại
    user_id: str
    user_roles: List[str]

    # Thông tin được các node điền vào
    query_analysis: Optional[QueryAnalysisOutput] = None # Kết quả phân tích câu hỏi
    final_answer: Optional[str] = None # Câu trả lời cuối cùng dưới dạng text
    clarification_needed: bool = False # Có cần hỏi lại người dùng không
    ask_clarification_questions: Optional[List[str]] = None # Các câu hỏi cần hỏi lại người dùng