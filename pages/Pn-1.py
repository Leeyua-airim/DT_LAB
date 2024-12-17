import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter


# 키워드 추출 함수 (한국어와 영어 지원)
def extract_keywords(text, top_n=5):
    """
    텍스트에서 한국어와 영어 키워드를 추출합니다.
    - text: 텍스트 (문자열)
    - top_n: 추출할 키워드 개수
    """
    # 특수문자 제거 및 단어 추출 (한국어, 영어, 숫자만 유지)
    clean_text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),  # 1-그램과 2-그램 추출
        stop_words="english"  # 영어 불용어 제거
    )
    X = vectorizer.fit_transform([clean_text])
    keywords = vectorizer.get_feature_names_out()
    return keywords[:top_n]


# Streamlit 앱 제목
st.title("Codestates & Rocket 느린 학습자를 위한 콘텐츠/문항 생성 APP")

# PDF 업로드 위젯
uploaded_file = st.file_uploader("교안을 업로드 해주세요. 확장자는 PDF 만을 지원합니다.", type=["pdf"])

if uploaded_file is not None:
    try:
        # 파일 저장 및 로드
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PyMuPDFLoader로 PDF 로드
        loader = PyMuPDFLoader("uploaded_file.pdf")
        doc = loader.load()

        # Text Splitter로 텍스트 분리
        test_splitter = SpacyTextSplitter(chunk_size=350, pipeline="ko_core_news_sm")
        split_docs = test_splitter.split_documents(doc)

        # 결과 출력
        st.subheader("키워드 추출 결과")
        if split_docs:
            all_keywords = Counter()  # 전체 키워드를 빈도로 저장

            for idx, split_doc in enumerate(split_docs):
                try:
                    # 각 청크에서 키워드 추출
                    keywords = extract_keywords(split_doc.page_content, top_n=5)
                    all_keywords.update(keywords)  # 키워드 빈도 누적

                    # 청크별 키워드 출력
                    st.write(f"청크 {idx + 1} 키워드: {', '.join(keywords)}")
                except Exception as e:
                    st.warning(f"청크 {idx + 1}에서 키워드 추출 중 문제가 발생했습니다: {str(e)}")

            # 전체 키워드 통합 및 정렬
            unique_keywords = [word for word, freq in all_keywords.most_common()]
            st.subheader("문서 전체에서 추출된 주요 키워드")
            st.write(", ".join(unique_keywords))
        else:
            st.info("PDF를 처리한 후 키워드가 여기에 표시됩니다.")

    except Exception as e:
        st.error("PDF 파일을 처리하는 중 문제가 발생했습니다. 오류: " + str(e))
else:
    st.info("PDF 파일을 업로드하면 처리된 내용이 여기에 표시됩니다.")
