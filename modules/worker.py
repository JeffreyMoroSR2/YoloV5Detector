from objects.source import Source
from libs.centroid.centroidtracker import CentroidTracker
from modules.loader import StreamLoader
from modules.analyzer import StreamAnalyzer


class Worker:
    def __init__(self, config, debugger):
        self.config = config
        self.debugger = debugger

        self.stream_loader = None
        self.stream_analyzer = None

        """Temp code"""
        streams = ['rtsp://admin:admin@192.168.1.2:554',
                   'rtsp://admin:admin@192.168.1.2:554']

        self.sources = []
        for i, stream in enumerate(streams):
            self.sources.append(Source(stream=stream, stream_id=i))
        """Temp code"""

    def start(self):
        self.stream_loader = StreamLoader(self.sources).start()
        self.stream_analyzer = StreamAnalyzer(stream_loader=self.stream_loader,
                                              trackers=[CentroidTracker() for _ in range(len(self.stream_loader))],
                                              config=self.config,
                                              debugger=self.debugger).start()
        return self

    def run(self):
        self.stream_analyzer.analyse()
