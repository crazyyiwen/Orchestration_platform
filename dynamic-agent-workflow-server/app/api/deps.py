from __future__ import annotations

from fastapi import Depends

from app.core.config import Settings, get_settings


def settings_dep() -> Settings:
    return get_settings()


SettingsDep = Depends(settings_dep)
