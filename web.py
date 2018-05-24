import json
import logging
import os
from wsgiref.simple_server import make_server

import falcon

from intent_extraction_serving_model import serve

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
model_api = serve()


def parse_headers(req_headers):
    headers_lst = ["CONTENT-TYPE", "CONTENT-ENCODING", "FORMAT",
                   "CLEAN", "DISPLAY-POST-PREPROCCES",
                   "DISPLAY-TOKENS", "DISPLAY-TOKEN-TEXT", "IS-HTML"]
    headers = {}
    for header_tag in headers_lst:
        if header_tag in req_headers:
            headers[header_tag] = req_headers[header_tag]
        else:
            headers[header_tag] = None
    return headers


def set_headers(res):
    """
    set expected headers for request (CORS)
    Args:
        res: the request
    """
    res.set_header('Access-Control-Allow-Origin', '*')
    res.set_header("Access-Control-Allow-Credentials", "true")
    res.set_header('Access-Control-Allow-Methods', "GET,HEAD,OPTIONS,POST,PUT")
    res.set_header('Access-Control-Allow-Headers',
                   "Access-Control-Allow-Headers, Access-Control-Allow-Origin,"
                   "Access-Control-Allow-Methods, Origin,Accept, X-Requested-With, Content-Type, "
                   "Access-Control-Request-Method, "
                   "Access-Control-Request-Headers, format, clean, "
                   "display-post-preprocces, display-tokens, "
                   "display-token-text, X-Auth-Token")


class IntentService(object):
    def on_options(self, req, res):
        res.status = falcon.HTTP_200
        set_headers(res)

    def on_get(self, req, resp):
        """Handles GET requests"""
        logger.info('handle GET request')
        resp.status = falcon.HTTP_200
        resp.body = resp.body = "Hello NER service"

    def on_post(self, req, resp):
        """Handles POST requests"""
        logger.info('handle POST request')
        resp.status = falcon.HTTP_200
        set_headers(resp)
        """ parse text ner

        custom parsing
        {
        'text': 'But Google is starting from behind.',
        'ents': [{'start': 4, 'end': 10, 'label': 'ORG'}],
        'title': None
        }
        """
        # input_json = json.loads(req.stream.read())
        text = req.media['text']
        # text = input_json['text']
        intent, entities = model_api(text)

        # doc = nlp(input_json['text'])
        # doc = nlp(text)
        # entities = displacy.parse_ents(doc)
        # for ent in entities['ents']:
        #     ent['type'] = ent.pop('label')
        entities['annotation_set'] = list(set([e['type'].lower() for e in entities['ents']]))
        entities['intent'] = intent
        resp.body = json.dumps(entities)


# falcon.API instances are callable WSGI apps
app = application = falcon.API()

ner_parser = IntentService()

app.req_options.auto_parse_form_urlencoded = True
app.add_route('/intent', ner_parser)
path = os.path.abspath(os.path.dirname(__file__))
app.add_static_route('/', path)

if __name__ == '__main__':
    port = 8000
    server = make_server('0.0.0.0', port, app)
    print('starting the server on port {0}'.format(port))
    server.serve_forever()
