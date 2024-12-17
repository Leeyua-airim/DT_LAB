import streamlit as st
import csv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
import openai
import requests
import ast

# Streamlit 페이지 설정
st.set_page_config(page_title="문항 생성기", layout="wide")


# Session State 초기화
if "generated_question" not in st.session_state:
    st.session_state["generated_question"] = None
if "news_keywords" not in st.session_state:
    st.session_state["news_keywords"] = None

# Chat 모델 초기화
chat = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.5,
    max_tokens=1500
)

# 교육 데이터
education_data = [
    {
        "id": 4,
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
    "input": "LLM 파인튜닝을 위해 학습용 데이터를 준비하는 과정을 평가할 수 있는 문항을 생성합니다.",
    "output": """
    [질문]
    A 기업은 특정 도메인에서 LLM(Large Language Model)을 파인튜닝하기 위해 학습 데이터를 준비하고 있다.
    다음 중 학습 데이터 준비 과정에서 가장 먼저 수행해야 하는 작업은 무엇인가?
    
    [선지]
    선지 1) 학습 데이터를 JSON 또는 CSV 형식으로 변환한다.
    선지 2) 데이터의 중복성을 제거하여 품질을 높인다.
    선지 3) 도메인 전문가의 피드백을 받아 레이블을 추가한다.
    선지 4) 학습에 필요한 데이터를 수집하여 전처리를 수행한다.

    [정답 및 해설]
    정답) 4
    해설) 학습 데이터 준비 과정에서 가장 먼저 수행해야 하는 작업은 데이터를 수집하고 전처리를 통해 모델 학습에 적합한 형태로 만드는 것입니다.
    """
    },
    {
    "input": "프롬프트 엔지니어링과 LLM 파인튜닝의 차이점을 평가할 수 있는 문항을 생성합니다.",
    "output": """
    [질문]
    A 기업은 LLM(Large Language Model)을 활용한 프로젝트에서 프롬프트 엔지니어링과 LLM 파인튜닝 중 적합한 방법을 선택해야 한다.
    다음 중 두 방법의 차이를 가장 정확히 설명한 것은 무엇인가?
    
    [선지]
    선지 1) 프롬프트 엔지니어링은 모델을 직접 수정하는 것이며, 파인튜닝은 외부 데이터를 사용하는 것이다.
    선지 2) 프롬프트 엔지니어링은 모델 학습 없이 질문 구조를 설계하는 것이며, 파인튜닝은 모델을 추가 데이터로 재학습시키는 것이다.
    선지 3) 프롬프트 엔지니어링은 데이터 레이블링을 포함하고, 파인튜닝은 데이터 레이블링이 필요하지 않다.
    선지 4) 프롬프트 엔지니어링과 파인튜닝은 모두 동일한 결과를 도출한다.

    [정답 및 해설]
    정답) 2
    해설) 프롬프트 엔지니어링은 모델을 수정하지 않고 입력을 최적화하는 방식이며, 파인튜닝은 모델을 특정 도메인 데이터로 재학습하는 과정입니다.
    """
    },
    {
    "input": "LLM 파인튜닝에 필요한 데이터 구조 설계를 평가할 수 있는 문항을 생성합니다.",
    "output": """
    [질문]
    A 기업은 LLM 파인튜닝에 사용할 데이터를 준비하고 있다. 다음 중 모델 학습에 적합한 데이터 구조는 무엇인가?
    
    [선지]
    선지 1) 데이터가 비정형 텍스트로 구성된 단순 파일
    선지 2) JSONL 포맷으로 각 데이터 샘플에 질문과 답변이 포함된 구조
    선지 3) Excel 파일로 정리된 분류 데이터
    선지 4) PDF 파일로 작성된 학습용 문서

    [정답 및 해설]
    정답) 2
    해설) JSONL은 각 줄에 JSON 객체를 포함하는 형식으로, 대규모 언어 모델 학습 데이터로 널리 사용됩니다.
    """
    }, 
    {
    "input": "LLM 파인튜닝 데이터 전처리를 평가할 수 있는 문항을 생성합니다.",
    "output": """
    [질문]
    LLM 파인튜닝을 위한 데이터 전처리 과정에서 가장 중요한 단계는 무엇인가?
    
    [선지]
    선지 1) 데이터에서 의미 없는 단어를 제거하여 데이터 양을 줄인다.
    선지 2) 모델 학습을 위해 데이터의 포맷을 JSONL로 변환한다.
    선지 3) 데이터에서 결측값을 확인하고 보완한다.
    선지 4) 데이터 샘플의 길이를 줄여 학습 속도를 높인다.
    
    [정답 및 해설]
    정답) 2
    해설) 데이터 전처리에서 중요한 단계 중 하나는 학습에 적합한 포맷으로 데이터를 변환하는 것입니다.
    """
    }, 
    {
    "input": "학습 데이터 레이블링 과정에서 고려해야 할 사항을 평가할 수 있는 문항을 생성합니다.",
    "output": """
    [질문]
    B 기업은 학습 데이터를 레이블링하여 LLM 파인튜닝에 활용하려고 한다.
    다음 중 학습 데이터 레이블링 과정에서 가장 중요한 고려사항은 무엇인가?
    
    [선지]
    선지 1) 레이블링 작업이 일관성 있게 수행되었는지 확인한다.
    선지 2) 데이터 샘플의 크기를 줄여 작업 시간을 단축한다.
    선지 3) 다양한 레이블을 추가하여 데이터의 복잡성을 높인다.
    선지 4) 레이블링 후 데이터를 암호화하여 보안성을 강화한다.
    
    [정답 및 해설]
    정답) 1
    해설) 데이터 레이블링 과정에서 가장 중요한 것은 일관성을 유지하는 것입니다. 이는 모델 학습 결과의 품질에 큰 영향을 미칩니다.
    """
    }
]



# Streamlit UI 구성
st.title("DTLAB 생성형 AI (P4_3_1) 지문 생성기(내부 PoC용)")
st.subheader(":rocket: [Step 1] 문항 생성을 위해 필요한 정보를 입력 후 지문을 생성합니다.", divider="gray")

# 이미지 데이터
image_card = {
    4: "/Users/ijaehwa/langchain/DT_LAB/image_card/P4_card.png",  # ID에 따라 이미지 매핑
}

# 사이드바 구성
st.sidebar.subheader("학습 맵(요인묶음)")
for i, data in enumerate(education_data):
    with st.sidebar.expander(f"P4-3-1(생성형 AI)"):
        # 이미지 표시
        if data["id"] in image_card:
            st.sidebar.image(
                image_card[data["id"]],
                use_container_width=True,
            )
        else:
            st.sidebar.write("이미지가 없습니다.")

        # 학습 맵 정보 표시
        st.sidebar.write(f"**대요인**: {data['factor']}")
        st.sidebar.write(f"**중요인**: {data['sub_factor']}")
        st.sidebar.write(f"**중요인 성취기준**: {data['achievement_standard']}")
        st.sidebar.write(f"**중요인에 대한 학습목표**: {data['learning_object']}")
        st.sidebar.write(f"**중요인의 문항 개발시 측정해야할 부분**: {data['learning_target_note']}")

# 기업 및 부서 리스트
companies = ["SK하이닉스", "코드스테이츠", "KCC", "현대모비스", "전기안전공사", "건강보험심사평가원"]
departments = ["데이터 엔지니어팀", "데이터 분석팀", "인공지능 연구팀", "디지털 마케팅팀", "DT 전략 기획팀", "교육 컨설팅 팀", "사업팀", "진단평가팀"]

selected_company = st.selectbox("📍현재 재직 중인 기업명을 선택하세요:", companies)
selected_department = st.selectbox("📍현재 소속 부서를 선택하세요:", departments)
employee_role = st.text_input("📍최근 주요하게 담당하고 계신 업무를 입력하세요:", placeholder="예: 사내 게시판 뉴스레터 작성 및 내용 검수")

# 키워드 상태 변수 초기화
if "news_keywords" not in st.session_state:
    st.session_state["news_keywords"] = None

# 주요 키워드 추출 버튼
if st.button("임직원 정보 기반 주요 키워드 추출"):
    if selected_company and selected_department and employee_role.strip():
        try:
            # Step 1: OpenAI를 사용해 키워드 추출
            st.warning("[안내] 직무 키워드 추출 중...")
            prompt = f"""
            다음 문장에서 중요한 키워드 3개를 추출 후 리스트로 반환하세요.

            문장: "{employee_role}"
            반환형식:["핵심단어1","핵심단어2","핵심단어3"]
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
            extracted_keywords_list = ast.literal_eval(extracted_keywords)
            formatted_keywords = " OR ".join([f'"{kw.strip()}"' for kw in extracted_keywords_list])

            # 키워드 저장
            st.session_state["news_keywords"] = formatted_keywords

            st.write("✅ 추출된 직무 키워드:")
            for i, keyword in enumerate(extracted_keywords_list, start=1):
                st.write(f"{i}. {keyword}")

            # Step 2: DeepSearch API를 통한 뉴스 검색
            st.warning("[안내] 뉴스 검색 중...")
            search_url = "https://api-v2.deepsearch.com/v1/articles"
            search_params = {
                "keyword": st.session_state["news_keywords"],
                "api_key": "e429ace02f9a48388882e71bd52ea740",
                "date_from": "2024-01-01",
                "date_to": "2024-11-15"
            }
            response = requests.get(search_url, params=search_params)

            if response.status_code == 200:
                data = response.json()
                articles = data.get("data", [])[:5]
                if articles:
                    st.write("🔍 관련 뉴스:")
                    for i, article in enumerate(articles, start=1):
                        st.write(f"{i}. {article.get('title', '제목 없음')}")

                    # Step 3: 뉴스에서 주요 키워드 추출
                    st.warning("[안내] 뉴스 키워드 요약 중...")
                    content = "\n\n".join(
                        f"제목: {article.get('title', '제목 없음')}\n요약: {article.get('summary', '요약 없음')}"
                        for article in articles
                    )
                    news_prompt = f"""
                    아래 기사 내용을 바탕으로 중요한 키워드 3개를 추출하세요:

                    {content}

                    반환형식: 키워드1, 키워드2, 키워드3
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
                    st.session_state["news_keywords"] = news_response["choices"][0]["message"]["content"].strip()
                    st.success("[뉴스 기반 키워드]")
                    st.write(st.session_state["news_keywords"])
                else:
                    st.warning("관련 기사를 찾을 수 없습니다.")
            else:
                st.error(f"뉴스 검색 실패: {response.status_code}")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
    else:
        st.warning("모든 입력값을 입력해주세요!")

# 난이도 선택
st.subheader(":rocket:[Step 2] 지문 생성 시 질문 난이도를 선택하세요.")
difficulty = st.radio("난이도 선택 : ", ("하", "중", "상"), index=2)

# 문항 생성 버튼
if st.button("문항 생성"):
    if st.session_state["news_keywords"]:
        st.warning("[안내] 문항 생성 중...")
        try:
            # 난이도에 따른 추가 설명 및 지문 길이 조정
            if difficulty == "하":
                complexity_instruction = "간결하고 기본적인 정보를 포함하여 짧은 지문을 생성하세요."
                max_tokens = 500
            elif difficulty == "중":
                complexity_instruction = "세부 정보를 포함하고 약간의 배경 설명을 추가하여 중간 길이의 지문을 생성하세요."
                max_tokens = 1000
            elif difficulty == "상":
                complexity_instruction = "심화된 설명과 추가적인 배경 정보를 포함하여 길고 상세한 지문을 생성하세요."
                max_tokens = 1500
            else:
                complexity_instruction = "기본 정보를 포함한 지문을 생성하세요."
                max_tokens = 500  # 기본값

            # 사용자 입력과 뉴스 키워드를 바탕으로 문제 설정
            user_input = (
                f"{selected_company} 기업의 {selected_department}에서 '{st.session_state['news_keywords']}' 주제를 다룹니다. "
                f"이와 관련하여 문제를 작성하세요."
            )

            # Prefix에 난이도와 추가 지시사항 포함
            prefix = (
                "당신은 시험문제를 생성하는 AI 비서입니다.\n"
                "제공된 정보를 바탕으로 객관식 문제를 생성하세요.\n\n"
                f"난이도: {difficulty}\n"
                f"추가 지시사항: {complexity_instruction}\n\n"
                "문항 형식:\n"
                "- 지문은 상황 설명을 포함하여 문제에 필요한 배경 정보를 제공합니다.\n"
                "- 질문은 객관식 형태로 작성됩니다. 예: '~중 가장 적절한 것은?', '~중 적절하지 않은 것은?'\n"
                "- 선택지는 4개를 제공하고, 하나는 정답이고 나머지는 오답으로 구성됩니다.\n\n"
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
                suffix=(
                    "Input: {input}\n"
                    "Output:\n"
                    "[질문]\n"
                    "{지문내용}\n\n"
                    "[선지]"
                    "선지 1) {선택지1}\n"
                    "선지 2) {선택지2}\n"
                    "선지 3) {선택지3}\n"
                    "선지 4) {선택지4}\n\n"
                    "[정답 및 해설]"
                    "정답) {정답}\n"
                    "해설) {해설}"
                ),
                input_variables=[
                    "input",  # 사용자 입력
                    "지문내용", "선택지1", "선택지2", "선택지3", "선택지4",  # 추가 변수 정의
                    "정답", "해설"
                ]
            )

            # 프롬프트 생성
            final_prompt = few_shot_prompt.format(
                input=user_input,
                지문내용="생성형 인공지능과 관련된 상황 설명을 포함하세요.",  # 지문 내용
                선택지1="선택지 1 내용",
                선택지2="선택지 2 내용",
                선택지3="선택지 3 내용",
                선택지4="선택지 4 내용",
                정답="올바른 선택지",
                해설="왜 이 선택지가 정답인지 설명하세요."
            )

            # Chat 모델 호출
            messages = [
                SystemMessage(content="당신은 시험 문제를 생성하는 AI 비서입니다."),
                SystemMessage(content=f"문제 생성 시 '{st.session_state['news_keywords']}' 주제를 포함합니다."),
                SystemMessage(content=complexity_instruction),
                HumanMessage(content=final_prompt)
            ]
            response = chat(messages)

            # 생성된 문항 저장
            st.session_state["generated_question"] = response.content
            st.success("문항 생성이 완료되었습니다.")
        except Exception as e:
            st.error(f"문항 생성 중 오류 발생: {str(e)}")
    else:
        st.warning("키워드 추출이 완료되지 않아 문항을 생성할 수 없습니다.")


# 생성된 문항 출력
if st.session_state["generated_question"]:
    st.subheader(":rocket:[Step 3] 문항 검수자 역할을 수행합니다.")
    
    # 생성된 문항 형식 분리
    generated_content = st.session_state["generated_question"]
    
    try:
        # 텍스트를 키워드로 분리
        question_content = ""
        options = ""
        correct_answer = ""
        explanation = ""

        # [질문] 부분 추출
        if "[질문]" in generated_content and "[선지]" in generated_content:
            question_content = generated_content.split("[질문]")[1].split("[선지]")[0].strip()
        
        # [선지] 부분 추출
        if "[선지]" in generated_content and "[정답 및 해설]" in generated_content:
            options = generated_content.split("[선지]")[1].split("[정답 및 해설]")[0].strip()

        # [정답 및 해설] 부분 추출
        if "[정답 및 해설]" in generated_content:
            answer_section = generated_content.split("[정답 및 해설]")[1].strip()
            if "정답)" in answer_section and "해설)" in answer_section:
                correct_answer = answer_section.split("정답)")[1].split("해설)")[0].strip()
                explanation = answer_section.split("해설)")[1].strip()
        
        # 분리된 내용 출력
        if question_content and options and correct_answer and explanation:
            st.write(f"**[질문]**\n{question_content}\n")
            st.write(f"**[선지]**\n{options}\n")
            st.write(f"**[정답]**\n{correct_answer}\n")
            st.write(f"**[해설]**\n{explanation}\n")
        else:
            st.warning("문항의 일부가 올바르게 생성되지 않았습니다. 원본 문항을 확인하세요.")
            st.write(generated_content)  # 원본 출력
    except Exception as e:
        st.error(f"문항을 분리하는 데 오류가 발생했습니다: {str(e)}")
        st.write(generated_content)  # 원본 출력


    # 문항 검수자 이름 입력
    reviewer_name = st.text_input("문항 검수자의 성함을 입력해주세요.", placeholder="예: 이재화")

    # CSV 저장 버튼
    if st.button("CSV로 저장"):
        if reviewer_name.strip():  # 검수자 이름이 입력되었는지 확인
            csv_filename = "generated_questions.csv"
            with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([question_content, options, correct_answer, explanation, reviewer_name])  # 세부 내용 저장
            st.success(f"문항과 검수자 정보가 {csv_filename} 파일에 저장되었습니다.")
        else:
            st.warning("검수자 이름을 입력해주세요.")

        # Session State 초기화
    for key in ["discarded_question", "discard_reason", "additional_reason"]:
        if key not in st.session_state:
            st.session_state[key] = None

    # 문항 폐기 버튼
    if st.button("AI resigns"):
        st.warning("문항 폐기 사유를 선택하고 추가 설명을 작성하세요.")
        st.caption("AI 는 실수할 수 있는 존재입니다. 학습데이터를 확보합니다.")
        st.session_state["discarded_question"] = True

    # 폐기 상태 활성화 시 폐기 입력 화면 표시
    if st.session_state["discarded_question"]:
        # 폐기 사유와 추가 설명 입력
        discard_reason = st.radio(
            "폐기 사유를 선택하세요:",
            ["질문 오류", "선지 오류", "정답 오류", "해설 오류", "기타"],
            key="discard_reason"
        )
        additional_reason = st.text_area(
            "폐기 사유에 대한 추가 설명을 작성해주세요. 없으면 공란으로 둡니다.",
            placeholder="예: 질문의 내용이 불명확하고, 선지가 모호합니다.",
            key="additional_reason"
        )
        reviewer_name = st.text_input("검수자의 성함을 입력해주세요.", placeholder="예: 이재화", key="reviewer_name")

        # 폐기 문항 저장 버튼
        if st.button("학습 데이터 저장"):
            if discard_reason and additional_reason.strip() and reviewer_name.strip():
                discarded_csv = "discarded_questions.csv"
                with open(discarded_csv, mode="a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        question_content,  # 질문
                        options,           # 선지
                        correct_answer,    # 정답
                        explanation,       # 해설
                        discard_reason,    # 폐기 사유
                        additional_reason, # 추가 설명
                        reviewer_name      # 검수자 이름
                    ])
                st.success(f"추가 학습용 문항과 폐기 사유가 {discarded_csv} 파일에 저장되었습니다.")
                st.session_state["discarded_question"] = None  # 상태 초기화
            else:
                st.warning("검수자 이름과 추가 설명을 모두 작성해주세요!")
