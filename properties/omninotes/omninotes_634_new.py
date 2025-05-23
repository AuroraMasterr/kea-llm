import sys
sys.path.append("..")
from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        if d(text="OK").exists():
            d(text="OK").click()
            
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/next").click()
        
        d(resourceId="it.feio.android.omninotes:id/done").click()
        
        if d(text="OK").exists():
            d(text="OK").click()

    @mainPath()
    def rule_remove_tag_from_note_shouldnot_affect_content_mainpath(self):
        d(resourceId="it.feio.android.omninotes:id/fab_expand_menu_button").long_click()
        d(resourceId="it.feio.android.omninotes:id/detail_content").set_text("#Hello")
        d(resourceId="it.feio.android.omninotes:id/detail_title").set_text("Hello22")
        d(description="drawer open").click()
        d(resourceId="it.feio.android.omninotes:id/note_title").click()

    @precondition(lambda self: d(resourceId="it.feio.android.omninotes:id/menu_attachment").exists()
                   and d(resourceId="it.feio.android.omninotes:id/menu_share").exists() and 
                   d(resourceId="it.feio.android.omninotes:id/menu_tag").exists() and
                   "#" in d(resourceId="it.feio.android.omninotes:id/detail_content").info["text"]
                   )
    @rule()
    def rule_remove_tag_from_note_shouldnot_affect_content(self):
        
        origin_content = d(resourceId="it.feio.android.omninotes:id/detail_content").info["text"]
        print("origin_content: " + str(origin_content))
        
        d(resourceId="it.feio.android.omninotes:id/menu_tag").click()
        
        if not d(className="android.widget.CheckBox").exists():
            print("no tag in tag list")
            return
        tag_list_count = int(d(className="android.widget.CheckBox").count)
        #tag_list_count = int(d(resourceId="it.feio.android.omninotes:id/md_control").count)
        tagged_notes = []
        for i in range(tag_list_count):
            # if d(resourceId="it.feio.android.omninotes:id/md_control")[i].info["checked"]:
            if d(className="android.widget.CheckBox")[i].info["checked"]:
                tagged_notes.append(i)
        if len(tagged_notes) == 0:
            print("no tag selected in tag list, random select one")
            selected_note_number = random.randint(0, tag_list_count - 1)
            d(className="android.widget.CheckBox")[selected_note_number].click()
            
            return
        selected_tag_number = random.choice(tagged_notes)
        select_tag_box = d(resourceId="it.feio.android.omninotes:id/md_control")[selected_tag_number]
        select_tag_name = select_tag_box.right(resourceId="it.feio.android.omninotes:id/md_title").info["text"].split(" ")[0]
        # select_tag_name = d(resourceId="it.feio.android.omninotes:id/title")[selected_tag_number+1].info["text"].split(" ")[0]
        print("selected_tag_number: " + str(selected_tag_number))
        print("selected_tag_name: " + str(select_tag_name))
        select_tag_name = "#"+select_tag_name
        
        select_tag_box.click()
        
        d(text="OK").click()
        

        assert not d(textContains=select_tag_name).exists()    
        new_content = d(resourceId="it.feio.android.omninotes:id/detail_content").info["text"].strip().replace("Content", "")
        print("new_content: " + str(new_content))
        origin_content_exlude_tag = origin_content.replace(select_tag_name, "").strip()
        print("origin_content_exlude_tag: " + str(origin_content_exlude_tag))
        
        assert new_content == origin_content_exlude_tag
    


if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/omninotes/OmniNotes-6.2.8.apk",
        device_serial="emulator-5554",
        output_dir="../output/omninotes/786/guided_new",
        policy_name="guided"
    )
    start_kea(t,setting)
    
