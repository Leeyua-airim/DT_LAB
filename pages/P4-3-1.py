import streamlit as st
import csv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import openai
import requests

# Streamlit 페이지 설정
st.set_page_config(page_title="문항 생성기", layout="wide")

# Chat 모델 초기화
chat = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.4,
    max_tokens=1500
)

# 교육 데이터
education_data = [
    {
        "factor": "생성형 인공지능",
        "sub_factor": "LLM 파인튜닝 역량",
        "achievement_standard": "생성형 인공지능 서비스를 특정 도메인 사용자 니즈에 맞게 활용하고자 LLM 파인튜닝에 필요한 학습용 데이터를 수집 후 전처리할 수 있다.",
        "learning_object": "LLM 파인튜닝과 이란 무엇인지 학습합니다. 프롬프트 엔지니어링 LLM 파인튜닝의 차이점에 대해 학습합니다. LLM 파인튜닝에 필요한 학습용 데이터의 파일(JSON, JSONL, CSV) 및 파일에 입력해야 하는 값은 무엇인지에 대하여 학습합니다.",
        "learning_target_note": "LLM 파인튜닝을 위해 자신의 업무에서 발생되어지는 데이터를 학습 데이터의 구조에 맞춰 전처리 할 수 있어야 하며, 문항 개발시 이를 측정할 수 있어야 한다."
    }
]

# Few-shot 예제
examples = [
    {
        "input": "시험문제의 지문 개발만을 진행합니다.",
        "output": (
            "A 기업의 데이터 분석팀은 현재 운영 중인 전통적인 머신러닝 시스템에서 생성형 AI 서비스로 대체하는 방안을 검토 중이다. "
            "다음 중 가장 적절하지 않은 접근 방법을 채택한 팀원은?"
        )
    },
]

# Streamlit UI 구성
st.title("DTLAB P4_3_1 지문 생성기(내부 PoC용)")
# st.write("현재는 지문만을 생성")
st.write("문항 생성을 위해 필요한 정보를 입력해주세요.")

# 교육 데이터 표시
st.sidebar.subheader("학습 맵(요인묶음)")
for i, data in enumerate(education_data):
        with st.sidebar.expander(f"P4-3-1(생성형 AI)"):
            st.write(f"**Factor**: {data['factor']}")
            st.write(f"**Sub Factor**: {data['sub_factor']}")
            st.write(f"**Achievement Standard**: {data['achievement_standard']}")
            st.write(f"**Learning Object**: {data['learning_object']}")
            st.write(f"**Learning Target Note**: {data['learning_target_note']}")


# 기업 리스트
companies = [
    "SK하이닉스",
    "코드스테이츠",
    "KCC",
    "현대모비스",
    "전기안전공사",
    "건강보험심사평가원"
]

# 부서 리스트
departments = [
    "데이터 엔지니어팀",
    "데이터 분석팀",
    "인공지능 연구팀",
    "디지털 마케팅팀",
    "DT 전략 기획팀",
    "교육 컨설팅 팀",
    "사업팀",
    "진단평가팀"
]


selected_company = st.selectbox("📍현재 재직 중인 기업명을 선택하세요:", companies)
selected_department = st.selectbox("📍현재 소속 부서를 선택하세요:", departments)


# 사용자 입력 받기
employee_role = st.text_input("📍최근 주요하게 담당하고 계신 업무를 입력하세요:", placeholder="예: 사내 게시판 뉴스레터 작성 및 내용 검수")


# 키워드 변수 초기화
news_keywords = None

# 주요 키워드 추출 버튼
if st.button("임직원 정보 기반 주요 키워드 추출"):
    if selected_company and selected_department and employee_role.strip():
        try:
            # Step 1: OpenAI를 사용해 `employee_role` 요약
            st.warning("[안내] 직무 키워드 추출 중...")
            prompt = f"""
            다음 문장에서 가장 중요한 2개의 핵심 단어를 추출하세요:

            "{employee_role}"

            핵심 단어 2개:
            """
            openai_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant for extracting keywords."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4
            )
            extracted_keywords = openai_response["choices"][0]["message"]["content"].strip()
            st.write(f"추출된 직무 키워드: {extracted_keywords}")

            # Step 2: DeepSearch API로 뉴스 검색
            st.warning("[안내] 직무 키워드 기반 뉴스를 검색 중...")
            formatted_keywords = " AND ".join([f'"{kw.strip()}"' for kw in extracted_keywords.split(",")])
            search_url = "https://api-v2.deepsearch.com/v1/articles"
            search_params = {
                "keyword": f'{formatted_keywords}',
                "api_key": "e429ace02f9a48388882e71bd52ea740",
                "date_from": "2024-06-01",
                "date_to": "2024-11-15"
            }
            response = requests.get(search_url, params=search_params)

            if response.status_code == 200:
                data = response.json()
                articles = data.get("data", [])[:5]
                if articles:
                    # 뉴스 기사 요약 생성
                    content = "\n\n".join(
                        f"제목: {article.get('title', '제목 없음')}\n요약: {article.get('summary', '요약 없음')}"
                        for article in articles
                    )
                    st.write("관련 뉴스:")
                    for i, article in enumerate(articles, start=1):
                        st.write(f"{i}. {article.get('title', '제목 없음')}")

                    # Step 3: 뉴스에서 주요 키워드 3개 추출
                    st.warning("[안내] 뉴스 키워드 요약 중...")
                    news_prompt = f"""
                    아래 기사 내용을 바탕으로 가장 중요한 키워드 3개를 추출하세요:

                    {content}

                    키워드 3개:
                    """
                    news_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an assistant for extracting keywords."},
                            {"role": "user", "content": news_prompt}
                        ],
                        max_tokens=200,
                        temperature=0.4
                    )
                    news_keywords = news_response["choices"][0]["message"]["content"].strip()
                    st.subheader("[임직원 정보에 기반한 뉴스 검색] -> 뉴스가 갖는 주요 키워드 추출")
                    st.write(news_keywords)
                else:
                    st.warning("관련 기사를 찾을 수 없습니다.")
            else:
                st.error(f"뉴스 검색 실패: {response.status_code}")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
    else:
        st.warning("모든 입력값을 입력해주세요!")

# 난이도 선택
difficulty = st.radio(
    "지문 생성 시 질문 난이도를 선택하세요:",
    ("하", "중", "상"),
    index=1
)

default_question = "LLM 파인튜닝 역량을 확인할 수 있는 지문을 생성합니다."

# 고정된 user_input과 f-string 결합

user_input = f"{selected_company} 기업에서 {news_keywords} 주제를 대상으로 하는 {default_question} (난이도: {difficulty})"


# Session State를 사용해 문항 상태 저장
if "generated_question" not in st.session_state:
    st.session_state["generated_question"] = None  # 초기화


# 문항 생성 버튼
if st.button("문항 생성"):
    if user_input:
        st.warning("[안내] 문항 내 지문 생성 중...")
        try:
            # Prefix에 교육 데이터 및 사용자 정보 포함
            prefix = (
                "당신은 역량 평가에 필요한 질문을 생성하는 임무를 맡은 AI 비서입니다.\n"
                "제공된 Education Data, 임직원 정보를 참고하여 지문을 생성하세요.\n\n"
                "Education Data:\n"
                f"- Factor: {education_data[0]['factor']}\n"
                f"- Sub Factor: {education_data[0]['sub_factor']}\n"
                f"- Achievement Standard: {education_data[0]['achievement_standard']}\n"
                f"- Learning Object: {education_data[0]['learning_object']}\n"
                f"- Learning Target Note: {education_data[0]['learning_target_note']}\n\n"
            )
            # FewShotPromptTemplate 정의
            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=PromptTemplate(
                    input_variables=["input", "output"],
                    template="Input: {input}\nOutput: {output}\n"
                ),
                prefix=prefix,
                suffix="Input: {input}\nOutput:",
                input_variables=["input"]
            )

            # 프롬프트 생성
            final_prompt = few_shot_prompt.format(input=user_input)
            messages = [
                SystemMessage(content="당신은 역량 평가에 필요한 질문을 생성하는 임무를 맡은 AI 비서입니다."),
                SystemMessage(content=f"질문의 난이도는 '{difficulty}'로 설정됩니다."),
                HumanMessage(content=final_prompt)
            ]
            # Chat 모델 호출
            response = chat(messages)
            st.session_state["generated_question"] = response.content
            st.success("문항 생성이 완료되었습니다.")
        except Exception as e:
            st.error(f"문항 생성 중 오류 발생: {str(e)}")
    else:
        st.warning("필요한 정보가 부족하여 문항을 생성할 수 없습니다.")

# 생성된 문항 출력
if st.session_state["generated_question"]:
    st.subheader("생성된 지문:")
    st.write(st.session_state["generated_question"])

    # 문항 검수자 이름 입력
    reviewer_name = st.text_input("문항 검수자의 성함을 입력해주세요.", placeholder="예: 이재화")

    # CSV 저장 버튼
    if st.button("CSV로 저장"):
        if reviewer_name.strip():  # 검수자 이름이 입력되었는지 확인
            csv_filename = "generated_questions.csv"
            with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([st.session_state["generated_question"], reviewer_name])  # 문항과 검수자 이름 저장
            st.success(f"문항과 검수자 정보가 {csv_filename} 파일에 저장되었습니다.")
        else:
            st.warning("검수자 이름을 입력해주세요!")