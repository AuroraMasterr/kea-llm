import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
try:
    cred = credential.Credential("AKIDkOsAGA53vjV7byBC1iHhBa5ShomY8WvB", "84oNky3zjHh0BD3mLjvvjLlz1rY9bsws")

    # 实例化一个client选项，可选的，没有特殊需求可以跳过
    client = hunyuan_client.HunyuanClient(cred, "")

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = models.ChatCompletionsRequest()
    params = {
        "TopP": 1,
        "Temperature": 1,
        "Model": "hunyuan-turbo",
        "Stream": False,
        "Messages": [
            {
                "Role": "user",
                "Content": "tell me a joke"
            }
        ]
    }
    req.from_json_string(json.dumps(params))

    # 返回的resp是一个ChatCompletionsResponse的实例，与请求对象对应
    resp = client.ChatCompletions(req)
    # 输出json格式的字符串回包
    print(resp.Choices[0].Message.Content)



except TencentCloudSDKException as err:
    print(err)