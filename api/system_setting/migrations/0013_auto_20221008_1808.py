# Generated by Django 3.0.1 on 2022-10-08 18:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('system_setting', '0012_auto_20200506_1326'),
    ]

    operations = [
        migrations.AlterField(
            model_name='systemsetting',
            name='title',
            field=models.CharField(max_length=20, null=True, verbose_name='标题'),
        ),
    ]
