from __future__ import annotations
from typing import Dict
from fastapi import APIRouter, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_302_FOUND
from modules.utils import verify_password
from config import config

router = APIRouter()

def init_context(config: dict, templates_path: str):
    global cfg, templates
    cfg = config
    templates = Jinja2Templates(directory=templates_path)

@router.get('/login')
async def login_page(request: Request):
    return templates.TemplateResponse('login.html', {
        'request': request,
        'cfg': config,
    })

@router.post('/login')
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    for user in cfg.get('users', []):
        if user['username'] == username and verify_password(password, user['password']):
            request.session['user'] = {'name': username, 'role': user.get('role', 'viewer')}
            next_url = request.query_params.get('next', '/')
            return RedirectResponse(next_url, status_code=HTTP_302_FOUND)
    return templates.TemplateResponse('login.html', {
        'request': request,
        'error': 'Invalid credentials',
        'cfg': config,
    })

@router.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse('/login', status_code=HTTP_302_FOUND)
