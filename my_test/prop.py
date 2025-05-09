from kea import *

class CheckSearchBox(KeaTest):
    @precondition(lambda self: d(resourceId="it.feio.android.omninotes.alpha:id/search_src_text").exists())
    @rule()
    def search_box_should_exist_after_rotation(self):
        d.rotate('l')
        d.rotate('n')
        assert d(resourceId="it.feio.android.omninotes.alpha:id/search_src_text").exists()