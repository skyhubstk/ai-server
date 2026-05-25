from openai import OpenAI
import os

client=OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def summarize(data,pred):

    prompt=f'''
기업:{data['company']}
예측 영업이익:{pred}
매출 추세:{data['trend']}
부채 위험:{data['risk']}

초보자도 이해할 수 있게
3줄 요약
장점3개
위험3개
'''

    res=client.chat.completions.create(
      model='gpt-5.5',
      messages=[{'role':'user','content':prompt}]
    )

    return res.choices[0].message.content