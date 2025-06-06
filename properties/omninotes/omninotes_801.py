import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/done").click()
        
    @mainPath()
    def swipe_locked_note_mainpath(self):
        d(resourceId="it.feio.android.omninotes:id/fab_expand_menu_button").long_click()
        d(resourceId="it.feio.android.omninotes:id/detail_content").set_text("Hello world")
        d(description="More options").click()
        d(text="Lock").click()
        d(resourceId="it.feio.android.omninotes:id/password").set_text("1")
        d(resourceId="it.feio.android.omninotes:id/password_check").set_text("1")
        d(resourceId="it.feio.android.omninotes:id/question").set_text("1")
        d(resourceId="it.feio.android.omninotes:id/answer").set_text("1")
        d(resourceId="it.feio.android.omninotes:id/answer_check").set_text("1")
        d(scrollable=True).scroll.to(text="OK")
        d(text="OK").click()
        d.press("back")
        d.press("back")


    @precondition(lambda self: d(resourceId="it.feio.android.omninotes:id/note_title").exists() and d(text="Notes").exists() and not d(text="Settings").exists() and d(resourceId="it.feio.android.omninotes:id/lockedIcon").exists())
    @rule()
    def swipe_locked_note(self):
        
        selected_note = d(resourceId="it.feio.android.omninotes:id/lockedIcon").up(resourceId="it.feio.android.omninotes:id/note_title")
        selected_note_text = selected_note.get_text()
        print("selected_note_text: " + selected_note_text)
        
        selected_note.scroll.horiz.forward(steps=100)
        time.sleep(3)
        d.press("recent")
        
        d.press("back")
        
        d.press("back")
        
        assert d(text=selected_note_text).exists()



if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/omninotes/OmniNotes-6.0.5.apk",
        device_serial="emulator-5554",
        output_dir="../output/omninotes/801/guided",
        policy_name="guided"
    )
    start_kea(t,setting)
    
