from django import template

register = template.Library()

@register.filter
def get_value(obj, field_name):
    value = getattr(obj, field_name, None)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value