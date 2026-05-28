#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sorinpark/4-huggingface/blob/main/Amazon_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# 한글 폰트 설치
get_ipython().system('sudo apt-get install -y fonts-nanum')
get_ipython().system('fc-cache -fv')

# matplotlib 캐시 파일 직접 삭제
import shutil, matplotlib
shutil.rmtree(matplotlib.get_cachedir())

# 라이브러리 설치
get_ipython().system('pip install transformers datasets torch pandas matplotlib seaborn -q')

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import pipeline

# 한글 폰트 적용
plt.rcParams['font.family'] = 'NanumBarunGothic'
plt.rcParams['axes.unicode_minus'] = False

# GPU 확인
device = 0 if torch.cuda.is_available() else -1
print(f"사용 디바이스: {'GPU ✅' if device == 0 else 'CPU ⚠️'}")

# 폰트 확인
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist if 'Nanum' in f.name]
print("설치된 나눔 폰트:", fonts)


# In[ ]:


# fancyzhx/amazon_polarity: 아마존 실제 상품 리뷰 데이터셋
# 컬럼: label(0=부정, 1=긍정), title(제목), content(본문)
# 별점 1~2 → 부정(0), 별점 4~5 → 긍정(1), 별점 3은 제외됨

dataset = load_dataset("fancyzhx/amazon_polarity")

# test split에서 100개만 선택
subset = dataset["test"].select(range(100))

# 데이터셋 구조 확인
print("데이터셋 구조:", subset)
print("\n컬럼:", subset.column_names)

# 샘플 미리보기
print("\n[샘플 1]")
print("제목   :", subset[0]["title"])
print("본문   :", subset[0]["content"][:100], "...")
print("실제레이블:", "긍정 😊" if subset[0]["label"] == 1 else "부정 😞")


# In[ ]:


# 모델 1: 이진 감정 분류 (POSITIVE / NEGATIVE)
sentiment_pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
    max_length=512
)

# 모델 2: 세부 감정 분류 (28가지: joy, anger, sadness, admiration...)
emotion_pipe = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    device=device,
    truncation=True,
    max_length=512
)

print("✅ 파이프라인 로드 완료!")

# 쇼핑 리뷰 스타일로 빠른 테스트
test = "Great product! Fast shipping and exactly as described. Will buy again!"
print("\n[이진 분류]", sentiment_pipe(test))
print("[세부 감정]", emotion_pipe(test))


# In[ ]:


def apply_sentiment(batch):
    """title + content 합쳐서 감정 분석 적용"""

    # title과 content를 합쳐서 더 풍부한 입력 생성
    # 예: "Great product! | Fast delivery and well packaged."
    texts = [
        t + " | " + c[:400]   # content 너무 길면 512 토큰 초과 방지
        for t, c in zip(batch["title"], batch["content"])
    ]

    # 1) 이진 감정 분류
    sent_res = sentiment_pipe(texts, batch_size=16)
    batch["pred_label"] = [r["label"]   for r in sent_res]
    batch["pred_score"] = [round(r["score"], 4) for r in sent_res]

    # 2) 세부 감정 분류
    emo_res = emotion_pipe(texts, batch_size=16)
    batch["emotion"]       = [r["label"]   for r in emo_res]
    batch["emotion_score"] = [round(r["score"], 4) for r in emo_res]

    return batch

# map()으로 전체 서브셋에 일괄 적용
result_ds = subset.map(
    apply_sentiment,
    batched=True,
    batch_size=16,
    desc="🛒 상품 리뷰 감정 분석 중..."
)

print("✅ 완료! 추가된 컬럼:", ["pred_label", "pred_score", "emotion", "emotion_score"])


# In[ ]:


df=result_ds.to_pandas()
df["actual"]=df["label"].map({0:"NEGATIVE",1:"POSITIVE"})

df["correct"]=df["pred_label"]==df["actual"]
accuracy=df["correct"].mean()

print(f"정확도:{accuracy:.1%} ({df['correct'].sum()}/100)")
print(f"긍정 예측:{(df['pred_label']=='POSITIVE').sum()}개")
print(f"부정 예측:{(df['pred_label']=='NEGATIVE').sum()}개")

df["title_short"]=df["title"].str[:35]+"..."
print("\n", df[["title_short","actual","pred_label","pred_score","emotion","correct"]].head(10))


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Amazon 상품 리뷰 감정 분석 결과", fontsize=14, fontweight="bold")

# 1) 긍정/부정 파이 차트
cnt = df["pred_label"].value_counts()
axes[0].pie(cnt, labels=["긍정", "부정"],
             colors=["#1dd1a1", "#ff6b6b"], autopct="%1.1f%%")
axes[0].set_title("긍정 / 부정 비율")

# 2) 세부 감정 상위 8개 바차트
top_emo = df["emotion"].value_counts().head(8)
axes[1].barh(top_emo.index, top_emo.values,
             color=sns.color_palette("husl", 8))
axes[1].set_title("세부 감정 분포 (상위 8개)")
axes[1].set_xlabel("리뷰 수")

# 3) 예측 신뢰도(score) 히스토그램
axes[2].hist(df["pred_score"], bins=20,
             color="#54a0ff", edgecolor="white")
axes[2].set_title("예측 신뢰도 분포")
axes[2].set_xlabel("Confidence Score")
axes[2].set_ylabel("리뷰 수")

plt.tight_layout()
plt.show()


# In[ ]:


save_cols = ["title", "content", "actual",
             "pred_label", "pred_score",
             "emotion", "emotion_score", "correct"]
df[save_cols].to_csv("amazon_sentiment_results.csv", index=False)
print("✅ 저장 완료: amazon_sentiment_results.csv")

