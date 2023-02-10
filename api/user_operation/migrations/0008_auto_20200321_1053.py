# Generated by Django 3.0.2 on 2020-03-21 10:53

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dormitories', '0009_auto_20200321_1053'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user_operation', '0007_auto_20200315_1316'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='waterfeeslog',
            name='change_water',
        ),
        migrations.AddField(
            model_name='waterfeeslog',
            name='change_money',
            field=models.DecimalField(decimal_places=2, default=1, max_digits=5, verbose_name='改变金额(元)'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='electricityfeeslog',
            name='dormitory',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='electricity_fees_log_s_dormitory', to='dormitories.Dormitory', verbose_name='宿舍'),
        ),
        migrations.AlterField(
            model_name='electricityfeeslog',
            name='operator',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='electricity_fees_log_s_operator', to=settings.AUTH_USER_MODEL, verbose_name='操作人'),
        ),
        migrations.AlterField(
            model_name='waterfeeslog',
            name='dormitory',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='water_fees_log_s_dormitory', to='dormitories.Dormitory', verbose_name='宿舍'),
        ),
        migrations.AlterField(
            model_name='waterfeeslog',
            name='operator',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='water_fees_log_s_operator', to=settings.AUTH_USER_MODEL, verbose_name='操作人'),
        ),
    ]
