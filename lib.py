import wbdata
import redis
import json

r = redis.Redis()

def get_data(indicator):
    """
    get_data returns a worldbank indicator dataset and caches it in Redis.
    """

    key = 'indicator:'+indicator
    body = r.get(key)
    if body:
        return json.loads(body)

    print('fetching indicator {}'.format(indicator))
    data = wbdata.get_data(indicator)
    r.set(key, json.dumps(data))
    return data
