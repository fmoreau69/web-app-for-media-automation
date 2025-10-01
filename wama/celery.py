"""
https://docs.celeryproject.org/en/stable/django/first-steps-with-django.html
https://www.section.io/engineering-education/django-celery-tasks/
https://buildwithdjango.com/blog/post/celery-progress-bars/
https://github.com/czue/celery-progress
"""

# import eventlet
# eventlet.monkey_patch()

import os
from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'wama.settings')

app = Celery('wama')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.conf.worker_pool = "threads"
app.conf.worker_concurrency = 4
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')


# @app.task
# def do_work(self, list_of_work, progress_observer):
#     total_work_to_do = len(list_of_work)
#     for i, work_item in enumerate(list_of_work):
#         # do_work_item(work_item)
#         # tell the progress observer how many out of the total items we have processed
#         progress_observer.set_progress(i, total_work_to_do)
#     return 'work is complete'
