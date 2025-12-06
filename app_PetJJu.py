import os
import torch
import requests
import langchain_huggingface
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
from transformers import AutoModel
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import pydeck as pdk
from tiktoken import encoding_for_model
from tiktoken import get_encoding
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import folium
from streamlit_folium import st_folium
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')
import uuid
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(dotenv_path='/Users/minsun/Desktop/4-2/빅콘테스트/code/.env')

# Pinecone API 초기화
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# API 키가 존재하지 않을 경우 예외 처리
if not pinecone_api_key or not pinecone_environment:
    raise ValueError("Pinecone API 키 또는 환경 변수가 설정되지 않았습니다.")

# LLM 클래스 정의 - Gemini-1.5-flash API 호출
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API 키가 설정되지 않았습니다.")

index_name = os.getenv("INDEX_NAME")

# 모델 및 토크나이저 초기화
if 'model_initialized' not in st.session_state:
    logger.info("START")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        logger.info("Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")

    # Embedding 모델 설정
    model_name = "upskyy/bge-m3-korean"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # VectorStore 초기화
    vectorstore = Pinecone.from_existing_index(index_name, embeddings.embed_query)

    # LLM 클래스 정의 - Gemini-1.5-flash API 호출
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        top_p=0.7,
        top_k=50,
        api_key=gemini_api_key
    )

    st.session_state['model_initialized'] = True
    st.session_state['tokenizer'] = tokenizer
    st.session_state['embeddings'] = embeddings
    st.session_state['vectorstore'] = vectorstore
    st.session_state['llm'] = llm
else:
    tokenizer = st.session_state['tokenizer']
    embeddings = st.session_state['embeddings']
    vectorstore = st.session_state['vectorstore']
    llm = st.session_state['llm']
    

# Streamlit 페이지 설정
st.set_page_config(page_title='🍊 PetJJu !')

# Streamlit 상태 초기화 및 사용자 ID 설정
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = {}

if "user_llm_calls" not in st.session_state:
    st.session_state["user_llm_calls"] = {}

# user_id가 없는 경우에만 생성
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())  # 고유 사용자 ID 생성

user_id = st.session_state["user_id"]

# 'displayed_messages'가 없는 경우 초기화
if "displayed_messages" not in st.session_state:
    st.session_state["displayed_messages"] = set()


# 'conversation_history'에서 user_id 초기화 확인 및 추가
if user_id not in st.session_state["conversation_history"]:
    st.session_state["conversation_history"][user_id] = {"messages": [], "filters": {}}
    st.session_state["user_llm_calls"][user_id] = 0  # LLM 호출 횟수 초기화

    
# 음식 종류 필터링에 사용할 키워드
food_keywords = {
    "카페": ["카페", "카페,디저트", "차,커피", "과일,주스전문점", "커피"],
    "한식": ["한식", "향토음식", "돼지고기구이", "찌개,전골", "국밥", "24시뼈다귀탕", "감자탕", "야식", "순대,순댓국", "비빔밥", "해장국", "백반,가정식", "보리밥", "곰탕,설렁탕", "한정식", "갈비탕", "추어탕", "매운탕,해물탕", "닭볶음탕", "생선구이", "불닭", "주꾸미요리", "전,빈대떡", "닭", "백숙,삼계탕", "냉면", "순대,순대국", "쌈밥"],
    "일식": ["일식당", "초밥", "초밥,롤", "일식튀김,꼬치", "일본식라면", "우동,소바", "덮밥", "꼬치구이", "이자카야"],
    "중식": ["중식당", "딤섬,중식만두", "양꼬치", "마라탕"],
    "양식": ["양식", "이탈리아음식", "피자", "스테이크,립", "스테이크", "브런치", "스파게티,파스타전문", "프랑스음식", "프랑스요리"],
    "술": ["바(BAR)", "맥주,호프", "맥주,요리주점", "요리주점", "포장마차", "전통,민속주점", "술집", "민속주점", "호프", "주류"]
}

# 계절 및 월 필터링에 사용할 키워드
month_keywords = {
    '1월': '2023년 1월 기준', '2월': '2023년 2월 기준', '3월': '2023년 3월 기준', '4월': '2023년 4월 기준',
    '5월': '2023년 5월 기준', '6월': '2023년 6월 기준', '7월': '2023년 7월 기준', '8월': '2023년 8월 기준',
    '9월': '2023년 9월 기준', '10월': '2023년 10월 기준', '11월': '2023년 11월 기준', '12월': '2023년 12월 기준'
}

season_keywords = {
    '봄' : ['2023년 3월 기준', '2023년 4월 기준', '2023년 5월 기준'],
    '여름': ['2023년 6월 기준', '2023년 7월 기준', '2023년 8월 기준'],
    '가을': ['2023년 9월 기준', '2023년 10월 기준', '2023년 11월 기준'],
    '겨울': ['2023년 12월 기준', '2023년 1월 기준', '2023년 2월 기준']
}

def create_search_filter(query):

    if query is None:
        return {}, {}, {}, {}

    filters_food = {}
    filters_place = {}
    filters_all = {}
    filters_pet = {}

    # 음식 종류 필터링
    for key, values in food_keywords.items():
        if key in query:
            filters_food["음식종류"] = {"$in": values}
            filters_all["음식종류"] = {"$in": values}

    if any(word in query for word in ["술", "소주", "맥주", "막걸리", "와인", "한잔", "간술"]):
        filters_food["음식종류"] = {"$in": food_keywords["술"]}
        filters_all["음식종류"] = {"$in": food_keywords["술"]}

    if any(word in query for word in ["바닷가", "해변", "바다", "해산물", "회", "싱싱", "횟집", "조개"]):
        filters_food["음식종류"] = {"$in": ["생선회", "해물,생선요리", "조개요리", "조개구이", "게요리", "대게요리"]}
        filters_all["음식종류"] = {"$in": ["생선회", "해물,생선요리", "조개요리", "조개구이", "게요리", "대게요리"]}

    if any(word in query for word in ["비오는", "울적한", "쓸쓸한", "쌀쌀한", "흐린", "장마", "추운", "겨울"]):
        filters_food["음식종류"] = {"$in": ["전,빈대떡", "해장국", "찌개,전골", "감자탕", "순대,순댓국", "샤브샤브", "국밥"]}

    if any(word in query for word in ["반려동물", "강아지", "애완동물", "반려견"]):
        filters_food["반려동물"] = {"$eq": "반려동물 동반 가능"}
        filters_all["반려동물"] = {"$eq": "반려동물 동반 가능"} 

    if any(word in query for word in ["현지", "숨겨진", "현지인", "로컬", "숨은", "아는 사람만 아는"]):
        filters_food["현지인이용건수비중"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}
        filters_all["현지인이용건수비중"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}

    if any(word in query for word in ["저렴", "싼", "가성비"]):
        filters_food["이용금액구간"] = {"$in": ["하위 20% 이상", "하위 10% 이상"]}
        filters_all["이용금액구간"] = {"$in": ["하위 20% 이상", "하위 10% 이상"]}

    elif any(word in query for word in ["비싼", "고급", "분위기 있는", "분위기 좋은"]):
        filters_food["이용금액구간"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}
        filters_all["이용금액구간"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}

    # 분위기 관련 필터링 추가 (핫플, 인기 장소 등)
    if any(word in query for word in ["핫플", "인기", "트렌디", "인싸", "힙한", "명소", "유명", "갬성", "뜨는", "핫한"]):
        filters_food["이용건수구간"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}
        filters_food["주요 고객 연령대"] = {"$in": ["30대", "20대 이하"]}          
        filters_all["이용건수구간"] = {"$in": ["상위 20% 이상", "상위 10% 이상"]}
        filters_all["주요 고객 연령대"] = {"$in": ["30대", "20대 이하"]}          

    # place 관련 필터링 추가
    if any(word in query for word in ["가 볼 만한", "볼거리", "놀거리", "관광지", "들릴", "가봐야"]):
        filters_place["분류명"] = {"$in": ["테마파크", "공연전시", "레저관광", "자연", "기타"]}

    # 분위기 관련 필터링 추가 (핫플, 인기 장소 등)
    if any(word in query for word in ["핫플", "인기", "트렌디", "인싸", "힙한", "명소", "유명", "갬성", "뜨는", "핫한"]):
        filters_place["총 언급 횟수"] =  {"$eq": "매우 많음"}
        filters_place["분류명"] = {"$in": ["테마파크", "공연전시", "레저관광", "자연", "기타"]}

    # 자연 관련 필터링 추가
    if any(word in query for word in ["힐링", "휴식", "명소"]):
        filters_place["분류명"] = {"$eq": "자연"}

    if any(word in query for word in ["반려동물", "강아지", "애완동물", "반려견"]):
        # "반려동물 언급 횟수"가 0보다 큰 경우만 필터링
        filters_place["반려동물 언급 횟수"] = {"$gt": 0}

    # 반려동물
    if any(word in query for word in ["반려동물", "강아지", "애완동물", "반려견"]):
        filters_food["반려동물"] = {"$eq": "반려동물 동반 가능"}
        filters_place["반려동물 언급 횟수"] = {"$gt": 0}

    # 지역 필터링 추가
    if local_area:
        filters_food["지역"] = {"$eq": local_area}
        filters_place["지역"] = {"$eq": local_area}

    # 계절 및 달 필터링 - 'restaurant' 네임스페이스에만 적용
    for month, date in month_keywords.items():
        if month in query:
            filters_all['기준연월'] = {"$eq": date}

    for season, dates in season_keywords.items():
        if season in query:
            filters_all['기준연월'] = {"$in": dates}

    # pet 관련 필터링 추가
    if any(word in query for word in ["아파", "병원"]):
        filters_pet["분류명"] = {"$eq": "반려의료"}

    if any(word in query for word in ["간식", "용품", "미용", "샤워"]):
        filters_pet["분류명"] = {"$eq": "반려동물 서비스"}

    if "중형" in query:
        filters_pet["입장가능 반려동물 크기"] = {"$in": ["모두 가능", "해당없음", "소형/중형", "7kg 미만", "30kg 미만", "20Kg 이하 소형,중형", "20kg 이하"]}

    if "대형" in query:
        filters_pet["입장가능 반려동물 크기"] = {"$in": ["모두 가능", "해당없음"]}

    return filters_food, filters_place, filters_all, filters_pet

# 임베딩 생성 함수
def embed_function(query):
    # HuggingFace의 embed_query 메서드를 통해 쿼리를 임베딩 벡터로 변환
    query_embedding = embeddings.embed_query(query)
    return query_embedding

# 네임스페이스별 search_kwargs 생성 함수
def create_search_kwargs(namespace, filters, k=1):
    search_kwargs = {"k": k, "namespace": namespace}
    if filters:
        search_kwargs["filter"] = filters
    return search_kwargs

# 네임스페이스별 문서 검색 함수
def retrieve_documents(namespace, filters, query):
    search_kwargs = create_search_kwargs(namespace, filters)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    return retriever.get_relevant_documents(query)


# 반려동물 동반 필터만 적용하는 함수
def search_only_pet_friendly_for_namespace(namespace, query, k=1):
    filters_pet_friendly = {
        "반려동물": {"$eq": "반려동물 동반 가능"}
    }


    # 해당 네임스페이스에 대해 반려동물 필터만 적용하여 검색
    search_kwargs = create_search_kwargs(namespace, filters_pet_friendly, k)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    # 검색 결과 반환
    return retriever.get_relevant_documents(query)


# 필터 결합 함수
def merge_filters(previous_filters, new_filters):
    combined_filters = {**previous_filters, **new_filters}
    return combined_filters


# 네임스페이스별 필터 결합 함수
def combine_namespace_filters(previous_filters, filters_food, filters_place, filters_all, filters_pet):
    combined_filters_food = merge_filters(previous_filters.get("filters_food", {}), filters_food)
    combined_filters_place = merge_filters(previous_filters.get("filters_place", {}), filters_place)
    combined_filters_all = merge_filters(previous_filters.get("filters_all", {}), filters_all)
    combined_filters_pet = merge_filters(previous_filters.get("filters_pet", {}), filters_pet)

    return {
        "unique": combined_filters_food,
        "place": combined_filters_place,
        "restaurant": combined_filters_all,
        "pet": combined_filters_pet
    }


# 검색 및 반려동물 필터 적용 함수
def search_with_fallback(namespace, filters, query):
    result = retrieve_documents(namespace, filters, query)

    if not result:
        result = search_only_pet_friendly_for_namespace(namespace, query)

    return result


# 검색 수행 함수
def dynamic_search(query, previous_filters):
    # 새로운 질문에 따른 필터 생성
    filters_food, filters_place, filters_all, filters_pet = create_search_filter(query)

    # 네임스페이스별 필터 결합
    namespace_filters = combine_namespace_filters(previous_filters, filters_food, filters_place, filters_all, filters_pet)

    combined_results = []
    results_by_namespace = {}  # 네임스페이스별 결과 저장용

    # 네임스페이스별 검색을 병렬로 수행
    with ThreadPoolExecutor() as executor:
        future_to_namespace = {executor.submit(search_with_fallback, namespace, filters, query): namespace for namespace, filters in namespace_filters.items()}

        for future in future_to_namespace:
            namespace = future_to_namespace[future]
            result = future.result()

            # 네임스페이스별 결과를 combined_results에 결합
            combined_results.extend(result)
            results_by_namespace[namespace] = result

    return combined_results


# 필터링 조건 생성 및 결합 함수
def process_filters(question, previous_filters):
    filters_food, filters_place, filters_all, filters_pet = create_search_filter(question)
    combined_filters = {
        "filters_food": {**previous_filters.get("filters_food", {}), **filters_food},
        "filters_place": {**previous_filters.get("filters_place", {}), **filters_place},
        "filters_all": {**previous_filters.get("filters_all", {}), **filters_all},
        "filters_pet": {**previous_filters.get("filters_pet", {}), **filters_pet}
    }
    return combined_filters


# 시스템 프롬프트 설정
system_prompt = """
너는 제주도 반려동물 동반 여행 및 맛집 추천 전문가야. 사용자가 질문을 하면 음식 종류, 위치, 그리고 특별한 요구 사항을 파악하고, 그 정보가 없을 경우 반려동물 동반 가능 맛집을 중심으로 추천을 제공해.

1. 사용자가 음식 종류, 위치, 그리고 요청 사항을 명확히 제시하지 않을 경우, 반려동물과 함께 즐길 수 있는 제주도의 인기 맛집과 장소를 기반으로 기본 추천을 제공해.
2. 추천할 가게에 대한 정보는 음식 맛, 분위기, 추천 메뉴, 반려동물 동반 가능 여부, 그리고 반려동물을 위한 추가 서비스나 편의 시설을 포함해.
3. 이전 응답에서 추천한 맛집은 다시 추천하지 마. 새로운 반려동물 친화 맛집을 찾아 추천해.
4. 검색된 맛집이 여러 개일 경우, 두 개에서 세 개의 반려동물 동반 가능 맛집을 추천해줘. 하지만 검색된 맛집이 하나밖에 없으면 그 하나의 맛집만 추천해.
5. 추가적인 질문 없이 현재 주어진 정보 내에서 최대한 답변을 마무리해. LLM 호출은 최대 3번으로 제한되며, 두 번째 응답에서 추가 질문 없이 마무리할 수 있도록 해.
6. 추천하는 맛집 근처의 반려동물과 함께 즐길 수 있는 관광 명소나 활동이 있다면 함께 알려줘.
7. 사용자의 이전 질문을 기억하여 맞춤화된 추천을 제공해.
8. 응답은 따듯하고 친근한 어조로 작성해줘.
9. 모든 응답은 반려동물 동반 여행 및 식사에 초점을 맞춘 지식 기반이어야 하며, 지식과 다른 정보는 아예 생성하지마.

"""

# 토큰 계산 함수 - 전역 변수로 인코딩 객체를 저장
try:
    global_encoding = encoding_for_model("gemini-1.5-flash")
except KeyError:
    global_encoding = get_encoding("cl100k_base")

@lru_cache(maxsize=10000)
def cached_token_count(text):
    return len(global_encoding.encode(text))


# 메시지를 LLM 형식으로 변환하는 함수
def convert_messages_for_llm(messages):
    llm_messages = []
    for message in messages:
        if isinstance(message, dict):
            # dict 형식의 메시지 처리
            role = message.get("role", "user")
            content = message.get("content", "")
        else:
            # HumanMessage, AIMessage, SystemMessage 객체 처리
            role = "user" if isinstance(message, HumanMessage) else (
                "assistant" if isinstance(message, AIMessage) else "system"
            )
            content = message.content

        if role == "user":
            llm_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            llm_messages.append(AIMessage(content=content))
        elif role == "system":
            llm_messages.append(SystemMessage(content=content))
    return llm_messages


# 토큰 수 계산
def count_tokens(messages_or_text):
    if isinstance(messages_or_text, str):
        # 문자열의 경우
        tokens = messages_or_text.split()
        return len(tokens)
    elif isinstance(messages_or_text, list):
        # 메시지의 리스트인 경우
        return sum(len(message["content"].split()) for message in messages_or_text)
    else:
        raise TypeError("입력은 문자열이나 메시지의 리스트여야 합니다.")



response_cache = {}  # 사용자별 응답 캐시
user_llm_calls = {}  # 사용자별 LLM 호출 횟수

# 프롬프트 생성 함수
def create_prompt(system_prompt, previous_messages, question, context):
    previous_messages_text = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in previous_messages])
    return f"""
    {system_prompt}

    이전 대화 내용:
    {previous_messages_text}

    사용자가 질문한 내용:
    {question}

    검색된 문서의 내용:
    {context}

    이 정보를 바탕으로 질문에 대한 답변을 생성해 주세요.
    """

# 문서 축소 함수
def reduce_context(docs, previous_messages, system_prompt, question):
    while len(docs) > 0:
        docs = docs[:-1]  # 가장 유사도가 낮은 문서를 제거
        context = generate_context_from_docs(docs)
        prompt_with_context = create_prompt(system_prompt, previous_messages, question, context)
        total_prompt_tokens = count_tokens(prompt_with_context)
        if total_prompt_tokens <= 5000:
            return prompt_with_context, docs
    raise ValueError("입력 토큰 수가 5,000을 초과했습니다.")

# 컨텍스트 생성 함수
def generate_context_from_docs(docs):
    return "\n\n".join([
        "\n".join([f"{key}: {value}" for key, value in doc.metadata.items()])
        for doc in docs
    ])

# LLM 호출 함수
def call_llm(llm, prompt_with_context):
    response = llm.invoke([HumanMessage(content=prompt_with_context)])
    return response

# 사용자 메시지 추가 함수
def add_user_message_to_history(previous_history, question):
    previous_history["messages"].append({"role": "user", "content": question, "timestamp": datetime.now()})

# 어시스턴트 메시지 추가 함수
def add_assistant_message_to_history(previous_history, response):
    previous_history["messages"].append({"role": "assistant", "content": response.content, "timestamp": datetime.now()})

# LLM 쿼리 함수
def query_llm(user_id: str, question: str):
    # LLM 호출 횟수 초기화 및 증가
    if "user_llm_calls" not in st.session_state:
        st.session_state["user_llm_calls"] = {}

    if user_id not in st.session_state["user_llm_calls"]:
        st.session_state["user_llm_calls"][user_id] = 0

    # if st.session_state["user_llm_calls"][user_id] >= 3:
    #     st.session_state["map_locations"] = []
    #     st.session_state["response"] = ""  
    #     if user_id in response_cache:
    #         return response_cache[user_id]
    #     raise ValueError("한 번의 질의에 대한 LLM 호출 횟수가 3회를 초과했습니다. \n 'Start New Chat'을 눌러 새로운 User ID로 대화를 시작하세요! 😊")
    if st.session_state["user_llm_calls"][user_id] >= 3:
        # LLM 호출 횟수 초과 시 처리
        st.session_state["map_locations"] = []
        st.session_state["response"] = ""   
        #   캐시된 응답이 있으면 반환
        if user_id in response_cache:
            return response_cache[user_id]
    
        # 사용자에게 친화적인 메시지 출력
        st.write(
            "한 번의 질의에 대한 LLM 호출 횟수가 3회를 초과했습니다. \n'새로운 채팅 시작 (Start New Chat)'을 눌러 새로운 User ID로 대화를 시작하세요! 😊"
             )
        return

    # 대화 이력과 필터링 조건 가져오기
    previous_history = st.session_state.conversation_history.get(user_id, {"messages": [], "filters": {}})
    previous_filters = previous_history.get("filters", {})

    # 사용자 메시지를 대화 이력에 추가
    add_user_message_to_history(previous_history, question)

    # 필터링 처리
    combined_filters = process_filters(question, previous_filters)

    # 동적 검색 실행
    docs = dynamic_search(question, combined_filters)

    # 검색된 문서의 컨텍스트 생성
    context = generate_context_from_docs(docs)

    # 전체 프롬프트 생성
    prompt_with_context = create_prompt(system_prompt, previous_history["messages"], question, context)

    # 전체 토큰 수 계산
    total_prompt_tokens = count_tokens(prompt_with_context)
    max_tokens = 5000

    # 토큰 수가 5,000을 초과하면 문서 축소
    if total_prompt_tokens > max_tokens:
        prompt_with_context, docs = reduce_context(docs, previous_history["messages"], system_prompt, question)

    # LLM 호출
    response = call_llm(llm, prompt_with_context)
    st.session_state["user_llm_calls"][user_id] += 1  # 호출 횟수 증가

    # 어시스턴트 메시지를 대화 이력에 추가
    add_assistant_message_to_history(previous_history, response)

    # 필터링 조건 업데이트
    filters_food, filters_place, filters_all, filters_pet = create_search_filter(question)
    combined_filters = {
        "filters_food": {**previous_filters.get("filters_food", {}), **filters_food},
        "filters_place": {**previous_filters.get("filters_place", {}), **filters_place},
        "filters_all": {**previous_filters.get("filters_all", {}), **filters_all},
        "filters_pet": {**previous_filters.get("filters_pet", {}), **filters_pet}
    }
    previous_history["filters"] = combined_filters

    # 응답을 캐시에 저장
    response_cache[user_id] = response.content

    # 대화 이력 저장
    st.session_state.conversation_history[user_id] = previous_history
    logger.info(response.content)

    return response.content



# Sidebar에서 Start New Chat 버튼 클릭 시 세션 초기화
with st.sidebar:
    #제주 아일랜드 사진 
    st.image('https://velog.velcdn.com/images/sunny_ho/post/386ea890-72ab-4e7f-8e8f-6967484d0158/image.png', width=280)
    st.write("")

    st.subheader("🍊 유저 정보")
    st.write(f"**User ID**: {st.session_state['user_id']}")  # 고유한 user_id 표시
    st.write("새로운 대화를 시작하려면 아래 버튼을 클릭하세요 :)")
    
    if st.button("Start New Chat", key="start_new_chat_button"):
        # 새로운 user_id 생성 및 초기화
        new_user_id = str(uuid.uuid4())
        st.session_state["user_id"] = new_user_id
        st.session_state["conversation_history"][new_user_id] = {"messages": [], "filters": {}}
        st.session_state["user_llm_calls"][new_user_id] = 0
        st.session_state["displayed_messages"] = set()

        st.session_state["name_list"] = []
        st.session_state["address_list"] = []
        st.session_state["text_list"] = []
        
        st.session_state["user_question"] = ""
        st.session_state["last_processed_question"] = ""
        st.session_state["map_locations"] = []
        st.session_state["response"] = ""  # 이전 답변 초기화
        st.write("\" 새로운 대화가 시작되었습니다! \"")

    

    st.subheader("제가 중요하게 생각하는 건")
    st.subheader("한라산의 유무 정도예요.")

    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Radio 버튼을 일반 변수에 할당하고, label_visibility='collapsed'로 설정
    local_area_choice = st.radio('', ('제주시 맛집', '서귀포시 맛집'), key="local_area_choice", label_visibility='collapsed')
    st.write("")
    local_area = "제주시" if local_area_choice == "제주시 맛집" else "서귀포"

    st.subheader("우리 반려동물도 함께..🥹")
    st.markdown(
        """
        <style>
        .stRadio > label {
            display: none;
        }
        .stRadio > div {
            margin-top: -20px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    local_choice = st.radio('', ('🐶💞 같이가요!', '사람🧍만 가요'), key="local_choice", label_visibility='collapsed')
    st.write("")
    pet_friendly = local_choice == "🐶💞 같이가요!"

st.write("")

st.title(" 🐾 멍식가들의 맛집 전쟁 !")
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("제주 요리사, 만나러 떠나볼까요?")
st.write("")

st.write("\" 우리 강아지도 같이 들어갈 수 있을까? \"")
st.write("당신의 특별한 하루를, 반려동물에 맞춘 완벽한 여행으로 만들어 보세요!")
st.write("")

# 이미지 추가
image_path = 'https://velog.velcdn.com/images/sunny_ho/post/ded7e8c8-2501-4a1a-b985-10f2ddd3807b/image.png'
st.markdown(f"<div style='display: flex; justify-content: center;'><img src='{image_path}' alt='centered image' width='400px'></div>", unsafe_allow_html=True)
st.write("")


# API 키 불러오기
kakao_api_key = os.getenv('KAKAO_API_KEY')
if not kakao_api_key:
    raise ValueError("Kakao API Key가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 초기 상태 설정
if 'name_list' not in st.session_state:
    st.session_state['name_list'] = []
if 'address_list' not in st.session_state:
    st.session_state['address_list'] = []
if 'text_list' not in st.session_state:
    st.session_state['text_list'] = []
if 'map_locations' not in st.session_state:
    st.session_state['map_locations'] = []

# 위경도 변환 함수
def addr_to_lat_lon(addr):  # KAKAO RESTAPI 사용
    url = f'https://dapi.kakao.com/v2/local/search/address.json?query={addr}'
    headers = {"Authorization": "KakaoAK " + kakao_api_key}
    response = requests.get(url, headers=headers)
    result = response.json()
    if 'documents' in result and len(result['documents']) > 0:
        match_first = result['documents'][0]['address']
        return float(match_first['y']), float(match_first['x'])  # 위도(y), 경도(x)
    else:
        logger.info(f"Address not found: {addr}")
        return None


# 사용자 질문 입력 받기
user_question = st.chat_input("질문을 입력해주세요! :", key="user_question_input")
if user_question:
    st.session_state["user_question"] = user_question  # 입력된 질문을 세션 상태에 저장

# 새로운 질문이 있고, 이전에 처리되지 않은 경우에만 처리
if (
    st.session_state.get("user_question")
    and st.session_state["user_question"] != st.session_state.get("last_processed_question")
):
    # 리스트 초기화
    st.session_state['name_list'] = []
    st.session_state['address_list'] = []
    st.session_state['text_list'] = []
    st.session_state['map_locations'] = []

    # 새로운 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(st.session_state["user_question"])

    # 대화 이력에 사용자 메시지 추가
    user_message = {
        "role": "user",
        "content": st.session_state["user_question"],
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.conversation_history[user_id]["messages"].append(user_message)

    # 표시된 메시지로 추가
    message_id = f"user_{user_message['timestamp']}"
    st.session_state["displayed_messages"].add(message_id)

    # LLM 호출 및 응답 생성
    with st.spinner("최고의 답변을 위해 열심히 생각 중..."):
        try:
            response_text = query_llm(
                user_id, st.session_state["user_question"]
            )
        except ValueError as e:
            st.error(str(e))
            response_text = ""
    
        if response_text:
            # 불필요한 따옴표 제거
            response_text = response_text.replace('\\"', '').replace('"', '')

            # 응답을 상태에 저장
            st.session_state["response"] = response_text  # 응답을 세션 상태에 저장

            # Assistant 응답 출력
            with st.chat_message("assistant"):
                st.markdown(response_text, unsafe_allow_html=True)

            # 대화 이력에 Assistant 메시지 추가
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.conversation_history[user_id]["messages"].append(assistant_message)

            # 표시된 메시지로 추가
            message_id = f"assistant_{assistant_message['timestamp']}"
            st.session_state["displayed_messages"].add(message_id)

    # 대화 이력과 필터 저장
    st.session_state.conversation_history[user_id]["filters"] = {
        "pet_friendly": pet_friendly,
        "local_area": local_area,
    }

    # 위치 정보를 가져오는 함수 호출
    docs = dynamic_search(st.session_state["user_question"], st.session_state.conversation_history[user_id]["filters"])
    for doc in docs:
        # 메타데이터에서 주소 추출
        if '주소' in doc.metadata:
            address = doc.metadata['주소']
        elif '관광지주소' in doc.metadata:
            address = doc.metadata['관광지주소']
        else:
            address = '주소 없음'

        # 메타데이터에서 가게 이름 추출
        if '가게이름' in doc.metadata:
            name = doc.metadata['가게이름']
        elif '관광지명' in doc.metadata:
            name = doc.metadata['관광지명']
        else:
            name = '이름 없음'

        text = doc.page_content

        # 리스트에 추가
        st.session_state['name_list'].append(name)
        st.session_state['address_list'].append(address)
        st.session_state['text_list'].append(text)

    # 주소 리스트를 이용하여 좌표 리스트 생성
    for address in st.session_state['address_list']:
        coords = addr_to_lat_lon(address)
        if coords:
            st.session_state['map_locations'].append(coords)

    # 마지막으로 처리한 질문 업데이트
    st.session_state["last_processed_question"] = st.session_state["user_question"]

# 기존 대화 메시지 표시
messages = st.session_state.conversation_history[user_id]["messages"]
for message in messages:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    timestamp = message.get("timestamp", "")

    # 메시지의 고유 식별자 생성
    message_id = f"{role}_{timestamp}"

    # 이미 표시된 메시지는 건너뜁니다
    if message_id in st.session_state["displayed_messages"]:
        continue

    # 'system' 역할의 메시지는 출력하지 않음
    if role != 'system':
        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

    # 표시된 메시지로 추가
    st.session_state["displayed_messages"].add(message_id)

# **응답과 지도를 동시에 표시하기 위해 두 개의 컨테이너로 분리**
response_container = st.container()
map_container = st.container()

# 응답을 별도의 컨테이너에 표시
with response_container:
    if st.session_state.get("response"):
        st.markdown(st.session_state["response"], unsafe_allow_html=True)

# 지도 표시를 별도의 컨테이너에 표시
with map_container:
    if 'map_locations' in st.session_state and st.session_state['map_locations']:
        # 첫 번째 위치를 중심으로 지도 생성
        lal_lon = st.session_state['map_locations'][0]
        m = folium.Map(location=lal_lon, zoom_start=13)
        
        # 각 위치에 마커 추가
        for name, coords in zip(st.session_state['name_list'], st.session_state['map_locations']):
            folium.Marker(coords, popup=name, tooltip=name).add_to(m)
        
        # 지도 가운데 정렬 CSS 추가
        m.get_root().html.add_child(folium.Element(
            """
            <style>
                .folium-map { display: flex; justify-content: center; }
            </style>
            """
        ))
        
        st_folium(m, width=700, height=500)
        logger.info("찐 지도 그렸어어")
    else:
        # 기본 제주도 지도 표시
        m = folium.Map(location=[33.4996213, 126.5311884], zoom_start=10)
        folium.Marker(location=[33.4996213, 126.5311884], popup='제주도', tooltip='제주도').add_to(m)
        
        # 지도 가운데 정렬 CSS 추가
        m.get_root().html.add_child(folium.Element(
            """
            <style>
                .folium-map { display: flex; justify-content: center; }
            </style>
            """
        ))
        
        st_folium(m, width=700, height=500)
        logger.info("그냥 지도 그렸어")