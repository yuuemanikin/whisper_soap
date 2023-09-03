import whisper
import json
import deepl
import os
from os.path import join, dirname
from dotenv import load_dotenv
from python_settings import settings

model = whisper.load_model("small")
voice_data_path = "/Users/tomoshigki/Desktop/whisper_test/whisper_test_env/小ファイル診察.m4a"
# result = model.transcribe("準備したファイル名を指定") # 今回の記事ではtest.m4aを用います。
result = model.transcribe(voice_data_path, fp16=False)
print(result["text"])

# API Keyを環境変数から読み込む
load_dotenv('/Users/tomoshigki/Desktop/whisper_test/whisper_test_env/api_key.env')
deepl_key = os.environ.get("DEEPL_API_KEY") # 環境変数の値を代入
openai_key = os.environ.get("OPENAI_API_KEY") # 環境変数の値を代入

# 以下はChapgptに投げる際に一旦英語へ翻訳して、再度日本語に戻すためのコード

API_KEY = os.environ.get('DEEPL_API_KEY') # 自身の API キーを指定


# まずは日本語を英語へ

text = result['text']
source_lang = 'JA'
target_lang = 'EN-US'

# イニシャライズ
translator = deepl.Translator(API_KEY)

# 翻訳を実行
result_text = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)

# print すると翻訳後の文章が出力される
print(result_text)

# chatgpt APIを呼んでSOAPへ分類する
import openai


openai.api_key = os.environ.get('OPENAI_API_KEY')

# OpenAI ChatCompletionの呼び出し
# プロンプトを変数化しておく

sys_content = "you are the secretary of a brilliant physician. I am going to present you with a transcribed text of a conversation between a doctor and a patient. You are to classify the following from that conversation."
assis_content = 'List the symptoms and problems that the patient is complaining about as "S data". List the findings (observation data) that the physician is examining and verbalizing as "O data". List as "A" the details (assessment) that the physician is considering during the examination in order to make a diagnosis. List as "P" what the doctor has presented to the patient as a plan of treatment. *Conversations are just written sentences. Words spoken by the doctor or the patient are not distinguished by line breaks, spaces, etc. Please classify them by context.'

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'system', 'content': sys_content},
        {'role': 'user', 'content': str(result_text)},
        {'role': 'assistant', 'content': assis_content},
    ],
)


# OpenAIの応答からテキストを取得
text = response['choices'][0]['message']['content']

# 次いで英語を再度日本語へ
text = str(text)
source_lang = 'EN'
target_lang = 'JA'

# イニシャライズ
translator = deepl.Translator(API_KEY)

# 翻訳を実行
result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)

# print すると翻訳後の文章が出力される
print(result)
