def format_result(input_data):
    result = []
    try:
        
        for i in input_data:
            result.append({i['word']:i['entity']})
    except Exception as ex:
        print("unable to format the repsonse -", ex)

    return result