from django.urls import include, path, re_path
from studio import views as studio_views

urlpatterns = [
    # Media is served by our own range-capable view (works with DJANGO_DEBUG=0 too and allows video seeking).
    re_path(r'^media/(?P<path>.+)$', studio_views.serve_media, name='media'),
    path('', include('studio.urls')),
]
