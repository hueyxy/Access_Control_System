# Generated by Django 3.0.2 on 2020-03-04 21:43

import datetime
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('dormitories', '0007_auto_20200304_2143'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('user_operation', '0002_waterratelog_operator'),
    ]

    operations = [
        migrations.CreateModel(
            name='ElectricityFeesLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('mode', models.CharField(choices=[('add', '加'), ('sub', '减')], default='sub', max_length=3, verbose_name='操作方式')),
                ('change_money', models.DecimalField(decimal_places=2, max_digits=5, verbose_name='改变金额(元)')),
                ('add_time', models.DateTimeField(default=datetime.datetime.now, verbose_name='创建时间')),
                ('note', models.CharField(blank=True, max_length=100, null=True, verbose_name='备注')),
                ('dormitory', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='electricity_fees_log_dormitory', to='dormitories.Dormitory', verbose_name='宿舍')),
                ('operator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='electricity_fees_log_operator', to=settings.AUTH_USER_MODEL, verbose_name='操作人')),
            ],
            options={
                'verbose_name': '宿舍电费使用记录',
                'verbose_name_plural': '宿舍电费使用记录',
            },
        ),
    ]
