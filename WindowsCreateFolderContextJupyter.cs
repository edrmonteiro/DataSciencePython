RegistryKey _key = Registry.ClassesRoot.OpenSubKey("Folder\\Shell", true);
RegistryKey newkey = _key.CreateSubKey("Jupyter");
newkey.SetValue("AppliesTo", "under:T:");

RegistryKey subNewkey = newkey.CreateSubKey("Command");
subNewkey.SetValue("", "C:\ProgramData\Anaconda3\Scripts\jupyter notebook");
subNewkey.Close();

newkey.Close();
_key.Close();