from openai import OpenAI
import os

client = OpenAI(
    # API keys are different for the Singapore and Beijing regions. To obtain an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
    api_key=os.getenv("QWEN_API_KEY"),
    # The following base_url is for the Singapore region. If you use a model in the Beijing region, replace the base_url with: https://dashscope.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

completion = client.chat.completions.create(
    model="qwen3-vl-flash",  # This example uses qwen3-vl-plus. You can replace it with another model name as needed. For a list of models, see https://www.alibabacloud.com/help/en/model-studio/getting-started/models
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {
                    "type": "text", 
                    "text": """Analyze the provided image of the blue TransMilenio bus and detect all visible physical damages on the front-left side, including dents, paint peeling, scratches, cracks, and structural deformations.  
Specifically identify and localize:  
- Deep dents and metal deformation on the lower front bumper  
- Paint peeling and exposed undercoat around the bumper and fender  
- Cracks and spiderweb-like fissures in the paint near the auxiliary turn signal  
- Deformation of the plastic housing around the orange turn signal  
- Bent or warped fender/wheel arch area  
- Paint chipping and scuffing around the fleet number “Z10-4649”  
- Any other visible impact marks or surface damage  
For each detected damage, output a separate bounding box with coordinates normalized to [0.0, 0.0] (top-left) to [1.0, 1.0] (bottom-right), using this format:  
```
{
  "label": "description of damage",
  "bbox": [x_min, y_min, x_max, y_max]
}
```  
Only return the JSON list of bounding boxes — no explanations, no extra text.
"""
                },
            ],
        },
    ],
)
print(completion.choices[0].message.content)