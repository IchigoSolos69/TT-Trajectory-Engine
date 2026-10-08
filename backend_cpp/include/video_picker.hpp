#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shobjidl.h>
#include <filesystem>
#include <string>

struct VideoSelection {std::string path,error;};
inline VideoSelection selectLocalVideo() {
    VideoSelection result;
    HRESULT initialized=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    if(FAILED(initialized)){result.error="Cannot initialize the file picker";return result;}
    IFileOpenDialog* dialog=nullptr;
    HRESULT hr=CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&dialog));
    if(SUCCEEDED(hr)) {
        const COMDLG_FILTERSPEC types[]={{L"Game recordings",L"*.mp4;*.avi;*.mkv;*.mov;*.webm;*.m4v"}};
        dialog->SetFileTypes(1,types);dialog->SetTitle(L"Select a table tennis recording");
        dialog->SetOptions(FOS_FILEMUSTEXIST|FOS_PATHMUSTEXIST|FOS_FORCEFILESYSTEM|FOS_NOCHANGEDIR);
        hr=dialog->Show(nullptr);
        if(SUCCEEDED(hr)) {
            IShellItem* item=nullptr;hr=dialog->GetResult(&item);
            if(SUCCEEDED(hr)) {
                PWSTR path=nullptr;hr=item->GetDisplayName(SIGDN_FILESYSPATH,&path);
                if(SUCCEEDED(hr)){result.path=std::filesystem::path(path).u8string();CoTaskMemFree(path);}
                item->Release();
            }
        }
        dialog->Release();
    }
    if(FAILED(hr) && hr!=HRESULT_FROM_WIN32(ERROR_CANCELLED)) result.error="The recording picker could not open the selected file";
    CoUninitialize();return result;
}
