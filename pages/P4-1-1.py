import streamlit as st
import csv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Streamlit 페이지 설정
st.set_page_config(page_title="문항 생성기", layout="wide")

# Chat 모델 초기화
chat = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.4,
    max_tokens=500
)

# 교육 데이터
education_data = [
    {
        "factor": "생성형 인공지능",
        "sub_factor": "업무에 필요한 생성형 인공지능 서비스 선정 역량",
        "achievement_standard": "단계별로 명확히 정의된 업무를 수행하는데 필요한 인공지능 서비스를 찾아낼 수 있다.",
        "learning_object": "업무를 지원할 수 있는 다양한 생성형 인공지능 서비스 탐색 방법에 대하여 살펴봅니다.",
        "learning_target_note": "업무 지원을 위한 최적의 AI 서비스를 선정했는지 확인합니다."
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
st.title("DTLAB P4_1_1 지문 생성기(내부 PoC용)")
# st.write("현재는 지문만을 생성")
st.write("문항 생성을 위해 필요한 정보를 입력해주세요.")

# 교육 데이터 표시
st.sidebar.subheader("학습 맵(요인묶음)")
for i, data in enumerate(education_data):
    with st.sidebar.expander(f"P4-1-1(생성형 AI)"):
        st.write(f"**Factor**: {data['factor']}")
        st.write(f"**Sub Factor**: {data['sub_factor']}")
        st.write(f"**Achievement Standard**: {data['achievement_standard']}")
        st.write(f"**Learning Object**: {data['learning_object']}")
        st.write(f"**Learning Target Note**: {data['learning_target_note']}")

# 기업 리스트
companies = [
    "코드스테이츠",
    "KCC",
    "현대모비스",
    "전기안전공사",
    "건강보험심사평가원"
]

# 부서 리스트
departments = [
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

# 고정된 user_input과 f-string 결합
default_question = "자신의 업무에 적합한 생성형 인공지능 서비스를 선정할 수 있는 역량을 확인할 수 있는 지문을 생성합니다."
if selected_company.strip() and selected_department.strip() and employee_role.strip():
    user_input = f"{selected_company} 기업의 {selected_department} 에서 {employee_role}을(를) 담당하는 직원을 대상으로 {default_question}"
else:
    user_input = None

# Session State를 사용해 문항 상태 저장
if "generated_question" not in st.session_state:
    st.session_state.generated_question = None

# 문항 생성 버튼
if st.button("문항 생성"):
    if user_input:
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

        # ChatCompletion 메시지 리스트 생성
        messages = [
            SystemMessage(content="당신은 역량 평가에 필요한 질문을 생성하는 임무를 맡은 AI 비서입니다."),
            HumanMessage(content=final_prompt)
        ]

        # Chat 모델 호출
        response = chat(messages)
        st.session_state.generated_question = response.content  # 상태에 저장

# 생성된 문항 출력
if st.session_state.generated_question:
    st.subheader("생성된 문항:")
    st.write(st.session_state.generated_question)

    # CSV 저장 버튼
    if st.button("CSV로 저장"):
        csv_filename = "generated_questions.csv"
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([st.session_state.generated_question])  # 상태에 저장된 문항 사용
        st.success(f"문항이 {csv_filename} 파일에 저장되었습니다.")
