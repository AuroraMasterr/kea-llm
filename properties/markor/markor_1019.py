from kea import *

class Test(KeaTest):
    

    @initializer()
    def set_up(self):
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(resourceId="net.gsantner.markor:id/next").click()
        
        d(text="DONE").click()
        
        
        if d(text="OK").exists():
            d(text="OK").click()
        
    @mainPath()
    def rotate_device_should_not_change_the_title_mainpath(self):
        d(resourceId="net.gsantner.markor:id/ui__filesystem_item__title").click()

    @precondition(
        lambda self: 
        d(resourceId="net.gsantner.markor:id/ui__filesystem_item__description").exists() and 
        d(resourceId="net.gsantner.markor:id/ui__filesystem_item__description").get_text() != "/storage/emulated/0/Documents"
        )
    @rule()
    def rotate_device_should_not_change_the_title(self):
        title = d(resourceId="net.gsantner.markor:id/toolbar").child(className="android.widget.TextView").get_text()
        print("title: "+str(title))
        assert title != "markor", "title should not be markor"




if __name__ == "__main__":
    t = Test()
    
    setting = Setting(
        apk_path="./apk/markor/2.3.2.apk",
        device_serial="emulator-5554",
        output_dir="../output/markor/1019/guided",
        policy_name="guided"
    )
    start_kea(t,setting)
    
