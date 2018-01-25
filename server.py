import tornado.ioloop
import tornado.web
import tornado.httpserver
import os
import PIL
import io

from tornado.options import define, options
define("port", default=3001, help="run on the given port", type=int)

from query_naive import query_from_sentence, query_from_image

print('start to listening')

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
        ]
        settings = dict(
            template_path = os.path.join(os.path.dirname(__file__), "templates"),
            static_path = os.path.join(os.path.dirname(__file__), "static"),
            debug = True,
        )
        tornado.web.Application.__init__(self, handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render(
            "index.html", title="Archtecture"
        )

    def post(self):
        query_text = self.get_argument("query_text", "")
        if not query_text == "":
            data_pair, images = query_from_sentence(query_text)
            self.render("result_full.html", data_pair=data_pair, images=images)
        elif len(self.request.files) > 0:
            imgfile = self.request.files['query_image'][0]['body']
            pil_image = PIL.Image.open(io.BytesIO(imgfile))
            pil_image = pil_image.convert('RGB')
            data_pair, images = query_from_image(pil_image)
            self.render("result_full.html", data_pair=data_pair, images=images)
        else:
            self.write('failed to upload file!')


if __name__ == '__main__':
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
