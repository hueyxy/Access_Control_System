# Generated by Django 3.0.1 on 2022-10-12 20:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('system_setting', '0013_auto_20221008_1808'),
    ]

    operations = [
        migrations.AlterField(
            model_name='systemsetting',
            name='title',
            field=models.CharField(max_length=20, verbose_name='标题'),
        ),
    ]
