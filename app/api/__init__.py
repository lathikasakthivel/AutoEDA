#!/usr/bin/env python3
"""
AutoEDA API Package

This package provides the API blueprint and endpoints for the AutoEDA system.
"""

from flask import Blueprint

bp = Blueprint('api', __name__)

from app.api import routes
