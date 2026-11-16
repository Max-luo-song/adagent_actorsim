"""Keep the entire common boundary on the exact original render code path."""
import contextlib
from environment import install_trajectory as original_install

def install_trajectory(session,path,absolute_source_start_s):
    receipt=original_install(session,path,absolute_source_start_s)
    owner=session.vehicles;edited_context=owner.render_context
    @contextlib.contextmanager
    def render_context(frame):
        if frame<=owner.activation_frame:yield
        else:
            with edited_context(frame):yield
    owner.render_context=render_context
    receipt['common_boundary_uses_exact_original_renderer']=True
    return receipt
