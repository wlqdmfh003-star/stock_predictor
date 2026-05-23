import requests

APP_KEY    = "PSAwuDAelQMNYcFDzmnAZn3ColuPL2FEkark"
APP_SECRET = "sqlzduq+0okjuZtqX3hTCGbFStFA2zQBUcc0/EkSnvVQqwrW1vappUcACHPRTwgm+njc4ZLloQi5Mk2fNU/sUk+0f/skc6m7QOVoC+G9TuroWhnzCENhANQkbEzKAzcaeLGH9elg1RsZcJNvzByawiCEkWl/2yyTYyEu2MsVOjegVfqHlbw="
ACCOUNT    = "50173435"

# 1. 토큰 발급
url  = "https://openapivts.koreainvestment.com:29443/oauth2/tokenP"
body = {
    "grant_type": "client_credentials",
    "appkey":     APP_KEY,
    "appsecret":  APP_SECRET
}
res   = requests.post(url, json=body)
token = res.json().get("access_token")
print("✅ 토큰 발급 성공")

# 2. 삼성전자 현재가 조회
url2 = "https://openapivts.koreainvestment.com:29443/uapi/domestic-stock/v1/quotations/inquire-price"
headers = {
    "content-type":  "application/json",
    "authorization": f"Bearer {token}",
    "appkey":        APP_KEY,
    "appsecret":     APP_SECRET,
    "tr_id":         "FHKST01010100"
}
params = {
    "FID_COND_MRKT_DIV_CODE": "J",
    "FID_INPUT_ISCD":         "005930"  # 삼성전자
}
res2 = requests.get(url2, headers=headers, params=params)
data = res2.json()
print("삼성전자 현재가:", data["output"]["stck_prpr"], "원")
print("전일대비:", data["output"]["prdy_vrss"], "원")
print("등락률:", data["output"]["prdy_ctrt"], "%")