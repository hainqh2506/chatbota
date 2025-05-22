# app_streamlit.py
import streamlit as st
# import asyncio # Không cần thiết nếu dùng invoke() đồng bộ hoàn toàn
import logging
import uuid

try:
    from state import AmelaReactCompatibleAgentState
    from graph_builder import build_graph # Giả sử nó trả về app có invoke()
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"Lỗi import: {e}. Đảm bảo các file của project được đặt đúng chỗ.")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_graph_app():
    logger.info("Bắt đầu khởi tạo graph cho Streamlit app...")
    try:
        app = build_graph() # build_graph() trả về compiled_app
        logger.info("Graph đã được khởi tạo thành công cho Streamlit.")
        # Kiểm tra xem app có phương thức invoke không
        if not hasattr(app, 'invoke'):
            logger.error("Đối tượng graph được biên dịch không có phương thức 'invoke'.")
            st.error("Lỗi cấu hình: Graph không hỗ trợ gọi đồng bộ (invoke).")
            return None
        return app
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi khởi tạo graph: {e}", exc_info=True)
        st.error(f"Không thể khởi tạo chatbot: {e}")
        return None

graph_app = load_graph_app()

def get_response_from_graph():
    pass
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"Tạo session Streamlit mới với ID: {st.session_state.session_id}")

if "messages_ui" not in st.session_state:
    st.session_state.messages_ui = []

if "langgraph_messages" not in st.session_state:
    st.session_state.langgraph_messages = [] # Đây là lịch sử cho LangGraph

if "user_roles" not in st.session_state:
    st.session_state.user_roles = ["staff"]

st.title("💬 Amber - RAG Chatbot Nội bộ (Đồng bộ)")
st.caption(f"Session ID: {st.session_state.session_id}")

with st.sidebar:
    st.header("Cấu hình người dùng")
    selected_roles = st.multiselect(
        "Chọn vai trò của bạn:",
        options=["staff", "manager", "developer", "newbie", "guest"],
        default=st.session_state.user_roles
    )
    if selected_roles != st.session_state.user_roles:
        st.session_state.user_roles = selected_roles
        st.rerun()
    st.info(f"Đang chạy với vai trò: {', '.join(st.session_state.user_roles)}")

for msg_ui in st.session_state.messages_ui:
    with st.chat_message(msg_ui["role"]):
        st.markdown(msg_ui["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    if not graph_app:
        st.error("Chatbot chưa sẵn sàng do lỗi khởi tạo graph.")
        st.stop()

    st.session_state.messages_ui.append({"role": "user", "content": prompt})
    st.session_state.langgraph_messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_message = "Amber đang suy nghĩ... 🤔"
        message_placeholder.markdown(thinking_message)

        try:
            graph_input = {
                "messages": st.session_state.langgraph_messages, # Truyền toàn bộ lịch sử hiện tại
                "original_query": prompt,
                "user_id": st.session_state.session_id,
                "user_roles": st.session_state.user_roles,
            }
            
            logger.info(f"Gọi graph.invoke() với input cho thread_id '{st.session_state.session_id}': "
                        f"Query='{prompt}', Roles={st.session_state.user_roles}, "
                        f"Num LangGraph Msgs={len(st.session_state.langgraph_messages)}")

            # Gọi phiên bản đồng bộ của graph
            final_graph_state = graph_app.invoke(
                graph_input, 
                config={"configurable": {"thread_id": st.session_state.session_id}}
            )
            # Xử lý kết quả
            if final_graph_state and final_graph_state.get("messages"):
                ai_message_obj = final_graph_state["messages"][-1]
                if isinstance(ai_message_obj, AIMessage):
                    full_response = ai_message_obj.content
                    st.session_state.langgraph_messages = final_graph_state["messages"] # Cập nhật lịch sử đầy đủ
                else:
                    full_response = "Lỗi: Phản hồi không đúng định dạng AIMessage."
                    logger.error(f"Phản hồi từ graph không phải AIMessage: {ai_message_obj}")
            elif final_graph_state: # Có state cuối nhưng không có messages hợp lệ
                 full_response = "Lỗi: State cuối từ graph không chứa AIMessage hợp lệ."
                 logger.error(f"State cuối không hợp lệ: {final_graph_state}")
            else: # Không có state cuối nào được trả về
                full_response = "Xin lỗi, Amber không thể xử lý yêu cầu của bạn lúc này (không có state cuối)."
                logger.error(f"Không nhận được state cuối từ graph.invoke: {final_graph_state}")


            message_placeholder.markdown(full_response)
            st.session_state.messages_ui.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"Lỗi khi gọi graph.invoke(): {e}", exc_info=True)
            error_msg = f"Đã xảy ra lỗi trong quá trình xử lý: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages_ui.append({"role": "assistant", "content": error_msg}) 