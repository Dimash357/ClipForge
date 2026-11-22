from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('projects/<int:pk>/', views.project, name='project'),
    path('history/', views.history, name='history'),
    path('system/', views.system, name='system'),
    path('api/projects/<int:pk>/status/', views.status, name='status'),
    path('api/projects/<int:pk>/cancel/', views.cancel, name='cancel'),
    path('api/projects/<int:pk>/resume/', views.resume, name='resume'),
    path('api/projects/<int:pk>/render/', views.render_clips, name='render_clips'),
    path('api/clips/<int:pk>/rerender/', views.rerender_clip, name='rerender_clip'),
    path('api/system/recheck/', views.system_recheck, name='system_recheck'),
    path('api/youtube/metadata/', views.metadata, name='metadata'),
    path('api/youtube/source-check/', views.source_check, name='source_check'),
]
