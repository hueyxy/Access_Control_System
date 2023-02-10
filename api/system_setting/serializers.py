from rest_framework import serializers

from .models import SystemSetting, SystemLog
import datetime


class SystemSettingSerializer(serializers.ModelSerializer):
    code = serializers.CharField(help_text="代码", max_length=20, allow_null=True)
    title = serializers.CharField(help_text="标题", max_length=20, allow_null=True)
    content = serializers.CharField(help_text="内容", max_length=100, allow_null=True)
    url = serializers.FileField(help_text="链接", allow_null=True)
    note = serializers.CharField(help_text="备注", max_length=100, allow_null=True)

    class Meta:
        model = SystemSetting
        fields = ("id", "code", "title", "content", "url", "note",)


class SystemSettingUpdateSerializer(serializers.ModelSerializer):
    water_fees = serializers.DecimalField(max_digits=5, decimal_places=2, help_text="水费(元/吨)")
    electricity_fees = serializers.DecimalField(max_digits=5, decimal_places=2, help_text="水费(元/吨)")
    temperature_upper = serializers.DecimalField(max_digits=5, decimal_places=2, help_text="体温上限(摄氏度)")
    temperature_lower = serializers.DecimalField(max_digits=5, decimal_places=2, help_text="体温下限(摄氏度)")
    todo_list = serializers.CharField(help_text="控制台代办事项列表")
    overview_info = serializers.CharField(help_text="控制台概略信息显示")
    data_overview_start_date = serializers.DateField(help_text="控制台数据概览起始日期", format="%Y-%m-%d")
    notice_title = serializers.CharField(help_text="首页公告标题")
    notice_content = serializers.CharField(help_text="首页公告内容")
    normal_access_start_time = serializers.TimeField(help_text="门禁正常通过开始时间", format="%H:%M:%S",
                                                     default="05-00-00")
    normal_access_end_time = serializers.TimeField(help_text="门禁正常通过截止时间", format="%H:%M:%S",
                                                   default="23-59-58")

    def validate_overview_info(self, overview_info):
        if len(overview_info.split(',')) != 4:
            raise serializers.ValidationError("概略信息显示只能选择四个选项！")
        return overview_info

    def validate_normal_access_end_time(self, normal_access_end_time):
        if self.initial_data["normal_access_start_time"] > self.initial_data["normal_access_end_time"]:
            raise serializers.ValidationError("门禁正常通过开始时间不能大于截止时间！")
        return normal_access_end_time

    def validate_temperature_upper(self, temperature_upper):
        if float(self.initial_data["temperature_upper"]) > 40.0:
            raise serializers.ValidationError("\n体温上限不能高于40.0度！")
        return temperature_upper

    def validate_temperature_lower(self, temperature_lower):
        if 35.0 > float(self.initial_data["temperature_lower"]):
            raise serializers.ValidationError("\n体温下限不能低于36.0度！")
        if float(self.initial_data["temperature_upper"]) < float(self.initial_data["temperature_lower"]):
            raise serializers.ValidationError("\n体温下限不能高于上限！")
        return temperature_lower

    class Meta:
        model = SystemSetting
        fields = (
            "water_fees", "electricity_fees", "temperature_upper", "temperature_lower", "todo_list", "overview_info",
            "data_overview_start_date", "notice_title",
            "notice_content", "normal_access_start_time", "normal_access_end_time",)


class SystemLogSerializer(serializers.ModelSerializer):
    id = serializers.IntegerField(help_text="ID")
    operator = serializers.SerializerMethodField(help_text="操作人")
    category = serializers.CharField(help_text="操作种类")
    content = serializers.CharField(help_text="操作内容")
    add_time = serializers.DateTimeField(help_text="操作时间", format="%Y-%m-%d %H:%M:%S")
    ip = serializers.CharField(help_text="操作IP")

    def get_operator(self, obj):
        return obj.operator.username + '(' + obj.operator.first_name + obj.operator.last_name + ')'

    class Meta:
        model = SystemLog
        fields = ("id", "operator", "category", "content", "add_time", "ip",)
