from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [('studio','0001_initial')]
    operations = [
        migrations.AddField(model_name='project',name='edit_mode',field=models.CharField(max_length=24,default='smart')),
        migrations.AddField(model_name='project',name='facecam_corner',field=models.CharField(max_length=16,default='auto')),
    ]
