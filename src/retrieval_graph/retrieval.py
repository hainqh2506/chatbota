import os
import logging
from typing import List, Dict, Any, Optional

from elasticsearch import Elasticsearch # Chỉ cần import Elasticsearch một lần
# from elasticsearch import helpers # Không dùng trong code này
from dotenv import load_dotenv
# Giả định VietnameseEmbeddings đã được import từ model_config
from configuration import VietnameseEmbeddings # Giữ lại import này

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Tải biến môi trường ---
load_dotenv()

# --- Tham số cho chiến lược (Có thể chuyển vào config file hoặc class init) ---
INITIAL_CANDIDATES_COUNT = 25
NUM_TOP_FILES_TO_CONSIDER = 4
NUM_TOP_CHUNKS_TO_CONSIDER = 4
FINAL_CONTEXT_COUNT = 6
RRF_K_CONST = 40
rrf_kw_weight = 0.3 # Trọng số cho KW trong RRF (có thể điều chỉnh)
rrf_vec_weight = 0.7 # Trọng số cho Vector trong RRF (có thể điều chỉnh)
# --- Cấu hình ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding") # Cung cấp giá trị mặc định
ELASTIC_URL = os.getenv("ELASTIC_URL")
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME", "amelav1") # Lấy từ env hoặc dùng mặc định

# --- Khởi tạo các thành phần toàn cục (Nếu dùng chung) ---
try:
    logging.info(f"Đang tải embedding model: {EMBEDDING_MODEL}...")
    embeddings = VietnameseEmbeddings(EMBEDDING_MODEL)
    # Lấy dimension nếu có thể (thay thế 1024 cứng)
    # embedding_dimension = getattr(embeddings, 'client', {}).get_sentence_embedding_dimension() # Ví dụ nếu model hỗ trợ
    embedding_dimension = 1024 # Tạm thời gán cứng nếu không lấy tự động được
    logging.info(f"Tải embedding model thành công. Dimension: {embedding_dimension}")
except Exception as e:
    logging.error(f"Lỗi nghiêm trọng: Không thể tải embedding model '{EMBEDDING_MODEL}'. Chi tiết: {e}", exc_info=True)
    exit(1)

try:
    logging.info(f"Đang kết nối tới Elasticsearch tại {ELASTIC_URL}...")
    es_client = Elasticsearch(
        hosts=ELASTIC_URL,
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        request_timeout=60,
        verify_certs=False, # Tắt nếu dùng self-signed cert, không khuyến nghị cho production
        ssl_show_warn=False,
        retry_on_timeout=True,
        max_retries=3 # Tăng retry
    )
    if not es_client.ping():
        raise ConnectionError("Kết nối tới Elasticsearch thất bại!")
    logging.info("Kết nối tới Elasticsearch thành công.")
except Exception as e:
    logging.error(f"Lỗi nghiêm trọng: Không thể kết nối tới Elasticsearch. Chi tiết: {e}", exc_info=True)
    exit(1)


class PQAThreeStageRetriever:
    """
    Retriever thực hiện tìm kiếm 3 giai đoạn trên Elasticsearch:
    1. Lấy ứng viên ban đầu (Hybrid Search: KW + KNN + ACL).
    2. Chọn file ứng viên từ kết quả bước 1.
    3. Tìm chunk tập trung trong các file đã chọn (Hybrid Search: KW + KNN + ACL).
    """
    def __init__(self, es_client: Elasticsearch, index_name: str, embeddings,
                 embedding_dimension: int = 1024, # Thêm embedding_dimension
                 initial_candidates_count: int = INITIAL_CANDIDATES_COUNT,
                 num_top_files: int = NUM_TOP_FILES_TO_CONSIDER,
                 num_top_chunks: int = NUM_TOP_CHUNKS_TO_CONSIDER,
                 final_context_count: int = FINAL_CONTEXT_COUNT,
                 rrf_k_const: int = RRF_K_CONST,
                 rrf_kw_weight: float = rrf_kw_weight,
                 rrf_vec_weight: float = rrf_vec_weight):
        """
        Khởi tạo Retriever.

        Args:
            es_client: Elasticsearch client đã khởi tạo.
            index_name: Tên index trong Elasticsearch.
            embeddings: Đối tượng embedding model (phải có phương thức embed_query).
            embedding_dimension: Số chiều của vector embedding.
            initial_candidates_count: Số lượng ứng viên ban đầu cần lấy.
            num_top_files: Số lượng file top đầu cần xem xét.
            num_top_chunks: Số lượng chunk top đầu có file chứa nó cần xem xét.
            final_context_count: Số lượng context cuối cùng trả về.
            rrf_k_const: Hằng số K cho công thức RRF.
        """
        if not isinstance(es_client, Elasticsearch):
            raise TypeError("es_client phải là một instance của Elasticsearch.")
        if not hasattr(embeddings, 'embed_query'):
            raise TypeError("Đối tượng embeddings phải có phương thức 'embed_query'.")

        self.es_client = es_client
        self.index_name = index_name
        self.embeddings = embeddings
        self.embedding_dimension = embedding_dimension
        self.initial_candidates_count = initial_candidates_count
        self.num_top_files = num_top_files
        self.num_top_chunks = num_top_chunks
        self.final_context_count = final_context_count
        self.rrf_k_const = rrf_k_const
        self.rrf_kw_weight = rrf_kw_weight
        self.rrf_vec_weight = rrf_vec_weight
        self.logger = logging.getLogger(__name__) # Sử dụng logger riêng cho class


    def _build_acl_filter(self, user_roles: List[str]) -> Dict[str, Any]:
        """Tạo phần filter ACL cho query Elasticsearch."""
        # Nếu không có role hoặc role là list rỗng, chỉ trả về các doc không có ACL
        if not user_roles:
            return {"bool": {"must_not": [{"exists": {"field": "access_control_list"}}]}}
        # Ngược lại, trả về các doc có role khớp HOẶC không có ACL
        return {"bool": {"should": [
                    {"terms": {"access_control_list": user_roles}},
                    {"bool": {"must_not": [{"exists": {"field": "access_control_list"}}]}}
                ], "minimum_should_match": 1}}

    def _combine_rrf(self, kw_results: Dict[str, int], vec_results: Dict[str, int]) -> Dict[str, float]:
        """Kết hợp kết quả từ KW và Vector search dùng RRF có trọng số."""
        combined_scores: Dict[str, float] = {}
        all_doc_ids = set(kw_results.keys()) | set(vec_results.keys())

        if not all_doc_ids:
            return {}

        # Lấy trọng số từ instance (ví dụ)
        kw_weight = getattr(self, 'rrf_kw_weight', 0.5) # Mặc định 0.5 nếu chưa set
        vec_weight = getattr(self, 'rrf_vec_weight', 0.5) # Mặc định 0.5 nếu chưa set
        k_const = self.rrf_k_const

        self.logger.info(f"Combining RRF with k={k_const}, kw_weight={kw_weight}, vec_weight={vec_weight}")


        for doc_id in all_doc_ids:
            rank_kw = kw_results.get(doc_id, float('inf'))
            rank_vec = vec_results.get(doc_id, float('inf'))

            score_kw = 1.0 / (k_const + rank_kw) if rank_kw != float('inf') else 0
            score_vec = 1.0 / (k_const + rank_vec) if rank_vec != float('inf') else 0

            # <<< Áp dụng trọng số >>>
            combined_scores[doc_id] = (score_kw * kw_weight) + (score_vec * vec_weight)

        return combined_scores


    def _get_initial_hybrid_candidates(self, user_query: str, user_roles: List[str]) -> List[Dict[str, Any]]:
        """
        Bước 1: Thực hiện hybrid search ban đầu (KW + KNN + ACL).
        """
        self.logger.info(f"--- Stage 1: Getting Initial Top {self.initial_candidates_count} Hybrid Candidates ---")
        query_vector = self.embeddings.embed_query(user_query)
        acl_filter = self._build_acl_filter(user_roles)

        # Lấy nhiều hơn để đảm bảo có đủ ứng viên sau RRF
        fetch_size = self.initial_candidates_count * 3
        knn_k = fetch_size
        knn_num_candidates = fetch_size * 2

        # --- Query 1: Keyword Search + ACL Filter ---
        keyword_results: Dict[str, int] = {}
        try:
            kw_query = {
                "size": fetch_size,
                "_source": False, # Chỉ cần _id
                "query": {
                    "bool": {
                        "must": [{
                            "bool": {
                                "should": [
                                    {"match": {"text_for_embedding": {"query": user_query, "boost": 1.0}}},
                                    {"match": {"metadata.doc_name": {"query": user_query, "boost": 1.2}}},
                                    {"match": {"metadata.keywords": {"query": user_query, "boost": 0.8}}}
                                ],
                                "minimum_should_match": 1
                            }
                        }],
                        "filter": [acl_filter] # Áp dụng ACL filter
                    }
                }
            }
            res_kw = self.es_client.search(index=self.index_name, body=kw_query, request_timeout=90)
            for rank, hit in enumerate(res_kw['hits']['hits'], 1):
                keyword_results[hit['_id']] = rank
            self.logger.info(f"  Keyword search (initial) found {len(keyword_results)} candidates.")
        except Exception as e:
            self.logger.error(f"  Error during initial keyword search: {e}", exc_info=True)

        # --- Query 2: Vector Search (KNN) + ACL Filter ---
        vector_results: Dict[str, int] = {}
        try:
            knn_query = {
                "size": fetch_size, 
                "_source": False,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector,
                    "k": knn_k,
                    "num_candidates": knn_num_candidates,
                    "filter": [acl_filter] # Áp dụng ACL filter
                }
            }
            # Kiểm tra xem có query_vector không trước khi search
            if query_vector:
                 res_vec = self.es_client.search(index=self.index_name, body=knn_query, request_timeout=90)
                 for rank, hit in enumerate(res_vec['hits']['hits'], 1):
                    vector_results[hit['_id']] = rank
                 self.logger.info(f"  Vector search (initial) found {len(vector_results)} candidates.")
            else:
                self.logger.warning("  Query vector is empty, skipping vector search.")

        except Exception as e:
            self.logger.error(f"  Error during initial vector search: {e}", exc_info=True)

        # --- Kết hợp RRF ---
        combined_scores = self._combine_rrf(keyword_results, vector_results)
        if not combined_scores:
            self.logger.warning("  No candidates found after combining KW and Vector search.")
            return []

        # Sắp xếp và lấy top IDs
        sorted_initial_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        top_initial_ids = [doc_id for doc_id, score in sorted_initial_docs[:self.initial_candidates_count]]
        self.logger.info(f"  Combined RRF resulted in {len(combined_scores)} candidates. Selecting top {len(top_initial_ids)}.")

        # --- Truy xuất Metadata (Level và Doc_ID) ---
        if not top_initial_ids:
            return []
        try:
            mget_resp = self.es_client.mget(
                index=self.index_name,
                body={"ids": top_initial_ids},
                _source_includes=["level", "doc_id"],
                request_timeout=60
            )
            initial_candidates = []
            id_to_metadata = {
                doc['_id']: {"level": doc['_source'].get('level'), "doc_id": doc['_source'].get('doc_id')}
                for doc in mget_resp.get('docs', []) if doc.get('found') and '_source' in doc
            }

            # Gắn metadata vào kết quả đã sắp xếp (duy trì thứ tự RRF)
            for doc_id, score in sorted_initial_docs:
                 if doc_id in top_initial_ids and doc_id in id_to_metadata: # Chỉ lấy những ID nằm trong top và có metadata
                     meta = id_to_metadata[doc_id]
                     initial_candidates.append({
                         "_id": doc_id,
                         "score": score, # Score RRF
                         "level": meta.get('level'),
                         "doc_id": meta.get('doc_id')
                     })
                 # Dừng nếu đã đủ số lượng mong muốn
                 if len(initial_candidates) >= self.initial_candidates_count:
                     break

            self.logger.info(f"  Retrieved metadata for {len(initial_candidates)} initial candidates.")
            return initial_candidates
        except Exception as e:
            self.logger.error(f"  Error retrieving metadata for initial candidates: {e}", exc_info=True)
            return []


    def _select_candidate_files(self, initial_candidates: List[Dict[str, Any]]) -> List[str]:
        """
        Bước 2: Từ danh sách ứng viên ban đầu, chọn ra các file ứng viên.
        """
        self.logger.info(f"--- Stage 2: Selecting Candidate Files (Top {self.num_top_files} Files, Top {self.num_top_chunks} Chunks) ---")
        top_file_ids = set()
        top_chunk_containing_file_ids = set()

        # Duyệt qua các ứng viên đã được sắp xếp theo RRF score từ bước 1
        for candidate in initial_candidates:
            doc_id_file = candidate.get('doc_id') # ID của file gốc (ví dụ: PR-AML-19-01)
            level = candidate.get('level')
            candidate_es_id = candidate.get('_id', 'Unknown ID') # Lấy _id từ candidate

            if not doc_id_file: # Bỏ qua nếu không có doc_id (không thể xác định file gốc)
                # SỬA Ở ĐÂY: Sử dụng candidate_es_id thay vì _id không xác định
                self.logger.warning(f"  Candidate {candidate_es_id} missing 'doc_id', skipping.")
                continue

            # Ưu tiên lấy ID file từ các document có level 'file' nằm trong top
            if level == 'file' and len(top_file_ids) < self.num_top_files:
                if doc_id_file not in top_file_ids: # Kiểm tra trùng lặp trước khi thêm
                    top_file_ids.add(doc_id_file)
                    self.logger.debug(f"  Added '{doc_id_file}' from a 'file' level candidate (ID: {candidate_es_id}).")

            # Lấy ID file từ các document chunk nằm trong top (nếu file đó chưa được chọn)
            elif level == 'content_chunk' and len(top_chunk_containing_file_ids) < self.num_top_chunks:
                 # Chỉ thêm nếu file này chưa nằm trong danh sách file từ chunk khác HOẶC từ level 'file'
                 if doc_id_file not in top_chunk_containing_file_ids and doc_id_file not in top_file_ids:
                    top_chunk_containing_file_ids.add(doc_id_file)
                    self.logger.debug(f"  Added '{doc_id_file}' from a 'content_chunk' level candidate (ID: {candidate_es_id}).")

            # Dừng sớm nếu đã thu thập đủ số lượng file từ cả hai nguồn
            # Logic dừng này có thể cần xem xét lại: Nếu cần đúng 3 file VÀ 3 chunk chứa file,
            # có thể bạn muốn đảm bảo cả hai điều kiện đều đạt đủ số lượng mong muốn, thay vì chỉ cần tổng số đạt.
            # Tuy nhiên, theo logic hiện tại là dừng khi cả hai đạt ngưỡng.
            if len(top_file_ids) >= self.num_top_files and len(top_chunk_containing_file_ids) >= self.num_top_chunks:
                 self.logger.debug("Reached target counts for both file and chunk-containing files. Stopping selection.")
                 break

        # Kết hợp cả hai set lại, loại bỏ trùng lặp tự động
        candidate_file_ids = top_file_ids.union(top_chunk_containing_file_ids)
        self.logger.info(f"  Selected {len(candidate_file_ids)} unique candidate file IDs: {candidate_file_ids}")
        return list(candidate_file_ids) # Trả về list


    def _find_focused_chunks(self, user_query: str, user_roles: List[str], candidate_file_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Bước 3: Thực hiện hybrid search cuối cùng, chỉ tìm chunk trong các file ứng viên.
        """
        if not candidate_file_ids:
            self.logger.warning("\n--- Stage 3: No candidate files selected, skipping final chunk search. ---")
            return []
        self.logger.info(f"\n--- Stage 3: Finding Top {self.final_context_count} Focused Chunks within {len(candidate_file_ids)} candidate files ---")
        self.logger.debug(f"  Searching within file IDs: {candidate_file_ids}")

        query_vector = self.embeddings.embed_query(user_query)
        acl_filter = self._build_acl_filter(user_roles)

        # Lấy nhiều hơn để RRF chính xác hơn
        fetch_size = self.final_context_count * 5
        knn_k = fetch_size
        knn_num_candidates = fetch_size * 2

        # --- Query 1: Keyword Search (Chunks only, filtered by doc_id and ACL) ---
        keyword_results_final: Dict[str, int] = {}
        try:
            kw_query_final = {
                 "size": fetch_size,
                 "_source": False,
                 "query": {
                     "bool": {
                         "must": [{ # Chỉ search trên text_for_embedding của chunk
                             "match": {"text_for_embedding": {"query": user_query, "boost": 1.0}}
                         }],
                         "filter": [
                             #{"term": {"level": "content_chunk"}},       # <<< Chỉ chunk >>>
                             {"terms": {"doc_id": candidate_file_ids}},  # <<< Trong file ứng viên >>>
                             acl_filter                                  # <<< ACL Filter >>>
                         ]
                     }
                 }
            }
            res_kw_final = self.es_client.search(index=self.index_name, body=kw_query_final, request_timeout=90)
            for rank, hit in enumerate(res_kw_final['hits']['hits'], 1):
                keyword_results_final[hit['_id']] = rank
            self.logger.info(f"  Keyword search (focused) found {len(keyword_results_final)} chunks.")
        except Exception as e:
            self.logger.error(f"  Error during focused keyword search: {e}", exc_info=True)

        # --- Query 2: Vector Search (Chunks only, filtered by doc_id and ACL) ---
        vector_results_final: Dict[str, int] = {}
        try:
            knn_query_final = {
                 "size": fetch_size, 
                 "_source": False,
                 "knn": {
                     "field": "embedding",
                     "query_vector": query_vector,
                     "k": knn_k,
                     "num_candidates": knn_num_candidates,
                     "filter": [ # <<< Các filter quan trọng >>>
                        # {"term": {"level": "content_chunk"}},      # Chỉ chunk
                         {"terms": {"doc_id": candidate_file_ids}}, # Trong file ứng viên
                         acl_filter                                 # ACL Filter
                     ]
                 }
            }
            if query_vector:
                res_vec_final = self.es_client.search(index=self.index_name, body=knn_query_final, request_timeout=90)
                for rank, hit in enumerate(res_vec_final['hits']['hits'], 1):
                    vector_results_final[hit['_id']] = rank
                self.logger.info(f"  Vector search (focused) found {len(vector_results_final)} chunks.")
            else:
                self.logger.warning("  Query vector is empty, skipping focused vector search.")
        except Exception as e:
            self.logger.error(f"  Error during focused vector search: {e}", exc_info=True)

        # --- Kết hợp RRF Final ---
        combined_scores_final = self._combine_rrf(keyword_results_final, vector_results_final)
        if not combined_scores_final:
            self.logger.warning("  No chunks found after combining focused KW and Vector search.")
            return []

        sorted_final_chunks = sorted(combined_scores_final.items(), key=lambda item: item[1], reverse=True)
        top_k_final_ids = [chunk_id for chunk_id, score in sorted_final_chunks[:self.final_context_count]]
        self.logger.info(f"  Combined focused RRF resulted in {len(combined_scores_final)} chunks. Selecting top {len(top_k_final_ids)}.")
        self.logger.debug(f"  Top {self.final_context_count} final chunk IDs: {top_k_final_ids}")

        # --- Truy xuất Nội dung cho Top K Final Chunks ---
        if not top_k_final_ids:
            return []
        try:
            mget_resp_final = self.es_client.mget(
                index=self.index_name,
                body={"ids": top_k_final_ids},
                _source_includes=[
                    "text_for_embedding", "metadata.doc_name", "metadata.source_url",
                    "metadata.page_number", "metadata.section_header", "level", "doc_id" # Thêm doc_id để tham chiếu
                ],
                request_timeout=60
            )
            final_contexts = []
            # Tạo dict để map ID với nội dung, giúp giữ đúng thứ tự RRF
            id_to_context = {}
            for doc in mget_resp_final.get('docs', []):
                if doc.get('found') and '_source' in doc:
                    _id = doc['_id']
                    source = doc['_source']
                    metadata = source.get('metadata', {})
                    context_text = source.get('text_for_embedding', '')

                    if context_text: # Chỉ thêm nếu có nội dung
                        # Lấy score RRF đã tính ở bước trước
                        rrf_score = combined_scores_final.get(_id)
                        if rrf_score is None:
                            self.logger.warning(f"Could not find RRF score for final candidate ID {_id}. Skipping.")
                            continue

                        context_info = {
                            "_id": _id, # Giữ lại ID gốc của chunk ES
                            "score": rrf_score,
                            "level": source.get('level'),
                            "doc_id": source.get('doc_id'), # ID file gốc
                            "doc_name": metadata.get('doc_name'),
                            "source_url": metadata.get('source_url'),
                            "page_number": metadata.get('page_number'),
                            "section_header": metadata.get('section_header'),
                            "context": context_text
                        }
                        id_to_context[_id] = context_info

            # Sắp xếp lại kết quả cuối cùng theo thứ tự ID đã được RRF xếp hạng
            ordered_contexts = [id_to_context[doc_id] for doc_id in top_k_final_ids if doc_id in id_to_context]

            self.logger.info(f"  Retrieved content for {len(ordered_contexts)} final chunks.")
            return ordered_contexts

        except Exception as e:
            self.logger.error(f"  Error retrieving final chunk documents: {e}", exc_info=True)
            return []


    def get_context(self, query: str, user_roles: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Thực hiện quy trình tìm kiếm 3 giai đoạn hoàn chỉnh.

        Args:
            query: Câu hỏi của người dùng.
            user_roles: List các role ID của người dùng để lọc ACL. Nếu None hoặc rỗng,
                        chỉ các tài liệu không có ACL sẽ được trả về.

        Returns:
            List các dictionary chứa context cuối cùng cho RAG, sắp xếp theo relevancy.
            Mỗi dict chứa: _id, score, level, doc_id, doc_name, source_url, page_number, section_header, context.
        """
        if user_roles is None:
            user_roles = [] # Đảm bảo là list rỗng nếu không có role

        self.logger.info(f"Starting 3-stage retrieval for query: '{query}' with roles: {user_roles}")

        # --- Bước 1: Lấy ứng viên ban đầu ---
        initial_candidates = self._get_initial_hybrid_candidates(query, user_roles)
        if not initial_candidates:
            self.logger.warning("Retrieval stopped: No initial candidates found.")
            return []

        # --- Bước 2: Chọn file ứng viên ---
        candidate_files = self._select_candidate_files(initial_candidates)
        if not candidate_files:
             self.logger.warning("Retrieval stopped: No candidate files selected.")
             return []

        # --- Bước 3: Tìm chunk tập trung ---
        final_contexts = self._find_focused_chunks(query, user_roles, candidate_files)

        if final_contexts:
            self.logger.info(f"--- Retrieval successful: Found {len(final_contexts)} final contexts for RAG ---")
        else:
             self.logger.warning("\nRetrieval finished: No relevant chunks found in the candidate files for the final stage.")

        return final_contexts


# --- Ví dụ sử dụng class ---
if __name__ == "__main__":
    # Khởi tạo Retriever với các thành phần đã tạo ở trên
    retriever = PQAThreeStageRetriever(
        es_client=es_client,
        index_name=INDEX_NAME,
        embeddings=embeddings,
        embedding_dimension=embedding_dimension, # Truyền dimension vào
        initial_candidates_count=INITIAL_CANDIDATES_COUNT,
        num_top_files=NUM_TOP_FILES_TO_CONSIDER,
        num_top_chunks=NUM_TOP_CHUNKS_TO_CONSIDER,
        final_context_count=FINAL_CONTEXT_COUNT,
        rrf_k_const=RRF_K_CONST
    )

    # Thực hiện tìm kiếm
    current_user_query = "CEO của Amela là ai?"
    #current_user_roles = ["role_staff", "group_all"] # Ví dụ có roles
    current_user_roles = [] # Ví dụ không có roles (chỉ lấy public docs)
    # current_user_roles = None # Tương tự list rỗng

    final_results = retriever.get_context(current_user_query, current_user_roles)

    # Xử lý kết quả (ví dụ: đưa vào LLM hoặc in ra)
    if final_results:
        logging.info("\n--- Final Contexts ---")
        for i, ctx in enumerate(final_results):
            logging.info(f"\nContext {i+1}:")
            logging.info(f"  ID (Chunk): {ctx['_id']}")
            logging.info(f"  Score (RRF): {ctx['score']:.6f}")
            logging.info(f"  Level: {ctx['level']}")
            logging.info(f"  Doc ID (File): {ctx['doc_id']}")
            logging.info(f"  Doc Name: {ctx['doc_name']}")
            logging.info(f"  Source URL: {ctx['source_url']}")
            if ctx['page_number']: logging.info(f"  Page: {ctx['page_number']}")
            if ctx['section_header']: logging.info(f"  Section: {ctx['section_header']}")
            logging.info(f"  Context Text Snippet: {ctx['context'][:200]}...") # In một phần context
    else:
        logging.warning("Không tìm thấy kết quả cuối cùng phù hợp.")