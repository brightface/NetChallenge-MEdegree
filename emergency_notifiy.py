import json
import sys
import requests


def send_slack_msg(message):
    url = "https://hooks.slack.com/services/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"
    title = "Emergency Situations :hospital:"  # 타이틀 입력
    slack_data = {
        "username": "Emergency Bot",  # 보내는 사람 이름
        "icon_emoji": ":hospital:",
        # "channel" : "#위급상황-알림",
        "attachments": [
            {
                "color": "#9733EE",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


if __name__ == "__main__":
    send_slack_msg("성인 남성 1명이 쓰러졌습니다.")
