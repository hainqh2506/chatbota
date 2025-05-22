# file: tools.py (tạo file mới)
import logging
from typing import List, Dict, Any, Optional, Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, create_react_agent
from state import AmelaReactCompatibleAgentState
#from retrieval import PQAThreeStageRetriever
logger = logging.getLogger(__name__)
try:
    # Giả định retriever.py cung cấp các thành phần này
    from retrieval import (
        PQAThreeStageRetriever,
        es_client,
        embeddings,
        embedding_dimension,
        INDEX_NAME as COLLECTION_NAME # Lấy tên index từ retriever.py
    )
    logger.info("Đã import thành công từ retriever.py")
except ImportError as e:
    logger.error(f"Lỗi import từ retriever.py: {e}. Đảm bảo file tồn tại và các biến được export.")
    exit(1)
except NameError as e:
     logger.error(f"Lỗi NameError khi import từ retriever.py: {e}. Biến có thể chưa được định nghĩa trong retriever.py.")
     exit(1)
try:
    # Các thành phần es_client, embeddings, embedding_dimension đã được import từ retriever.py
    retriever = PQAThreeStageRetriever(
        es_client=es_client,
        index_name=COLLECTION_NAME,
        embeddings=embeddings,
        embedding_dimension=embedding_dimension
    )
    logger.info(f"Đã khởi tạo PQAThreeStageRetriever cho index: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo PQAThreeStageRetriever: {e}", exc_info=True)
    exit(1)
def format_retriever_results(relevant_docs: List[Dict[str, Any]]) -> str:
    """Định dạng kết quả từ PQAThreeStageRetriever thành chuỗi cho LLM."""
    contexts = []
    if not relevant_docs:
        return "Không tìm thấy tài liệu nội bộ nào liên quan."
    logger.info(f"Đang định dạng {len(relevant_docs)} tài liệu từ retriever...")
    for i, doc in enumerate(relevant_docs):
        content = doc.get("context", "N/A").strip()
        doc_name = doc.get("doc_name", "N/A")
        source_url = doc.get("source_url")
        # page_number = doc.get("page_number")
        # section_header = doc.get("section_header")
        chunk_id = doc.get("_id", "N/A")

        metadata_lines = [f"--- Document {i+1} (ID: {chunk_id}) ---"]
        metadata_lines.append(f"Source Name: {doc_name}")
        if source_url: metadata_lines.append(f"Source URL: {source_url}")
        # if page_number is not None: metadata_lines.append(f"Page: {page_number}")
        # if section_header: metadata_lines.append(f"Section: {section_header}")

        formatted_chunk = "\n".join(metadata_lines) + f"\nContent:\n{content}\n"
        contexts.append(formatted_chunk)
        # logger.debug(f"Formatted context {i+1} for ID '{chunk_id}'") # Giữ debug nếu cần

    final_formatted_string = "\n".join(contexts) # Tạo chuỗi cuối cùng
    # logger.debug(f"Chuỗi context hoàn chỉnh sau khi định dạng:\n{final_formatted_string}") # Log debug toàn bộ chuỗi nếu cần
    return final_formatted_string
# --- Placeholder cho PQAThreeStageRetriever và các hàm liên quan ---
# Chúng ta sẽ tích hợp logic từ retriever.py và doc_search_tool.py vào đây sau.
# class PQAThreeStageRetrieverPlaceholder:
#     def __init__(self, es_client=None, index_name=None, embeddings=None, embedding_dimension=None):
#         logger.info("PQAThreeStageRetrieverPlaceholder initialized (chưa có logic thật).")
#         self.es_client = es_client
#         self.index_name = index_name
#         # ... và các tham số khác

#     def get_context(self, query: str, user_roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
#         logger.info(f"[PQA Placeholder] Searching for query: '{query}' with roles: {user_roles}")
#         # Logic giả:
#         if "quy trình nghỉ việc" in query.lower():
#             return [
#                 {
#                     "_id": "doc_nghiviec_123", "score": 0.9, "level": "content_chunk",
#                     "doc_id": "HR_POLICY_001", "doc_name": "Chinh Sach Nghi Viec Amela",
#                     "source_url": "http://sharepoint/hr/nghiviec", "page_number": 1,
#                     "section_header": "Quy trinh xin nghi viec",
#                     "context": "Để nghỉ việc, nhân viên cần nộp đơn xin nghỉ việc cho quản lý trực tiếp và phòng nhân sự trước 30 ngày..."
#                 }
#             ]
#         return []

# # Khởi tạo retriever placeholder (sẽ thay bằng retriever thật sau)
# retriever_placeholder = PQAThreeStageRetrieverPlaceholder()
def format_retriever_results_placeholder(relevant_docs: List[Dict[str, Any]]) -> str:
    if not relevant_docs:
        return "Không tìm thấy tài liệu nội bộ nào liên quan (placeholder)."
    contexts = []
    for i, doc in enumerate(relevant_docs):
        content = doc.get("context", "N/A").strip()
        doc_name = doc.get("doc_name", "N/A")
        contexts.append(f"--- Tài liệu {i+1}: {doc_name} ---\n{content}\n")
    return "\n".join(contexts)
# @tool
# def amela_documents_search_tool(query: str, user_roles: List[str]) -> str: # Giờ là hàm đồng bộ
#     """
#     Tìm kiếm tài liệu nội bộ Amela liên quan đến một câu hỏi (query) và danh sách vai trò người dùng (user_roles).
#     Bạn PHẢI cung cấp cả 'query' và 'user_roles'.
#     """
#     logger.info(f"[Tool] amela_documents_search_tool called with query: '{query}', user_roles from LLM: {user_roles}")
    
#     if user_roles is None: # Pydantic schema sẽ không cho phép điều này nếu không có default
#         logger.warning("  amela_documents_search_tool nhận user_roles là None, dùng list rỗng.")
#         effective_roles_for_search = []
#     else:
#         effective_roles_for_search = user_roles
            
#     # Giả sử retriever.get_context là đồng bộ
#     relevant_docs = retriever.get_context(query, effective_roles_for_search)
#     return format_retriever_results(relevant_docs)

@tool
def amela_documents_search_tool(query: str, state: Annotated[AmelaReactCompatibleAgentState, InjectedState]) -> str:
    """
    (Placeholder) Tìm kiếm tài liệu nội bộ Amela liên quan đến query.
    """
    qpa_output = state.get("query_analysis")
    # Ưu tiên dùng user_roles đã được QPA xử lý, hoặc asker_role_context nếu retriever của bạn hỗ trợ
    effective_roles_for_search = []
    if qpa_output and qpa_output.user_roles:
        effective_roles_for_search = qpa_output.user_roles
    elif state.get("user_roles"): # Fallback về user_roles ban đầu nếu QPA không có
        effective_roles_for_search = state.get("user_roles")
    else:
        effective_roles_for_search = []
    
    logger.info(f"[Tool] amela_documents_search_tool called with query: '{query}', effective_roles: {effective_roles_for_search}")
    relevant_docs = retriever.get_context(query, effective_roles_for_search)
    return format_retriever_results(relevant_docs)
@tool
async def company_structure_tool(query: str) -> str:
    """
    (Placeholder) Cung cấp thông tin về cơ cấu tổ chức, phòng ban, đội nhóm, tên viết tắt của Amela.
    """
    logger.info(f"[Tool Placeholder] company_structure_tool called with query: '{query}'")
    query_lower = query.lower()
    if "istqb" in query_lower:
        return "ISTQB là một chứng chỉ quốc tế về kiểm thử phần mềm (Placeholder)."
    elif "sơ đồ tổ chức" in query_lower or "phòng ban" in query_lower:
        return "Amela có nhiều phòng ban... (Placeholder). Vui lòng xem SharePoint."
    return f"Không tìm thấy thông tin cho '{query}' trong cơ cấu tổ chức (Placeholder)."

# Placeholder cho Google Search Tool nếu cần
# from langchain_community.tools import GoogleSearchAPIWrapper
# google_search_tool = GoogleSearchAPIWrapper() # Cần GOOGLE_API_KEY và GOOGLE_CSE_ID
# Hoặc một tool đơn giản:
@tool
async def google_search_placeholder_tool(query: str) -> str:
    """(Placeholder) Tìm kiếm trên Google."""
    logger.info(f"[Tool Placeholder] google_search_placeholder_tool called with query: '{query}'")
    return f"Kết quả tìm kiếm Google cho '{query}': [Nội dung từ Google - Placeholder]"

# Tập hợp các tools sẽ được sử dụng bởi Main Assistant
main_assistant_tools = [amela_documents_search_tool, company_structure_tool, google_search_placeholder_tool]