import json
import boto3
import asyncio
import functools

# Define the client to interact with AWS Lambda
client = boto3.client('lambda')

async def compare(arr, pdf_path, docx_path):
    loop = asyncio.get_running_loop()
    # print(pdf_path, docx_path)
    objects = await asyncio.gather(
        *[  
            loop.run_in_executor(None, functools.partial(client.invoke, FunctionName = 'dev-comparePdfDoc',InvocationType = 'RequestResponse',Payload = json.dumps({
                'pdf_prifix_path': pdf_path + i,
                'docx_prifix_path': docx_path + i
            })))
            for i in arr
        ]
    )
    return objects
 
def lambda_handler(event,context):
    print(event)
    body = json.loads(event['body'])
    pages = body['pages']
    pdf_path = body['pdf_path']
    docx_path = body['docx_path']
    
    # print(pages, pdf_path, docx_path)
    
    # pdf_path = 'images/sample_102_pdf_jpg/'
    # docx_path = 'images/sample_102_docx_jpg/'
    # pages = 2;

    arr = []
    i = 0
    while i < pages:
      arr.append(str(i+1) + '.jpg')
      i+=1

    # print(arr)
    result = []
    loop = asyncio.get_event_loop()
    objects = loop.run_until_complete(compare(arr, pdf_path, docx_path))
    
    for i in objects:
      result.append(json.loads(i['Payload'].read()))

    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },        
        'body': json.dumps({
            'result': result
        }),
	}
            
    # return result

   
    