import requests

def check_url(url):
    try:
        response = requests.get(url, timeout=5)
        # 如果响应状态码在200-400之间，那么我们认为URL是可以访问的
        if response.status_code >= 200 and response.status_code < 400:
            print(f"URL {url} is accessible.")
        else:
            print(f"URL {url} is not accessible. Status code: {response.status_code}")
    except requests.exceptions.RequestException as err:
        print (f"URL {url} is not accessible. Error: {err}")

# 测试
# check_url("http://phoenix0.d2.comp.nus.edu.sg:55570")
check_url("http://phoenix0.d2.comp.nus.edu.sg:55563")