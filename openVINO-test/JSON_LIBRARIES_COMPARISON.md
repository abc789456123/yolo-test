# jsoncpp vs nlohmann/json 비교

## 📊 주요 차이점 비교표

| 항목 | jsoncpp | nlohmann/json |
|------|---------|---------------|
| **설치 방법** | `apt install libjsoncpp-dev` | `apt install nlohmann-json3-dev` |
| **라이브러리 타입** | 컴파일 필요 (`-ljsoncpp`) | Header-only (링킹 불필요) |
| **Include** | `#include <jsoncpp/json/json.h>` | `#include <nlohmann/json.hpp>` |
| **네임스페이스** | `Json::Value`, `Json::arrayValue` | `nlohmann::json` 또는 `using json = nlohmann::json` |

## 🔍 코드 비교

### 1. JSON 객체 생성

**jsoncpp:**
```cpp
Json::Value metadata;
Json::Value objects(Json::arrayValue);
metadata["timestamp"] = "2025-09-10T10:21:52Z";
metadata["objects"] = objects;
```

**nlohmann/json:**
```cpp
json metadata;
metadata["timestamp"] = "2025-09-10T10:21:52Z";
metadata["objects"] = json::array();  // 또는 그냥 []
```

### 2. 중첩 객체 생성

**jsoncpp:**
```cpp
Json::Value obj;
obj["class_id"] = det.class_id;
obj["bbox"]["x"] = det.box.x;
obj["bbox"]["y"] = det.box.y;
objects.append(obj);
```

**nlohmann/json:**
```cpp
json obj = {
    {"class_id", det.class_id},
    {"bbox", {
        {"x", det.box.x},
        {"y", det.box.y}
    }}
};
metadata["objects"].push_back(obj);
```

### 3. JSON 문자열 변환

**jsoncpp:**
```cpp
Json::StreamWriterBuilder builder;
std::string json_string = Json::writeString(builder, metadata);
```

**nlohmann/json:**
```cpp
std::string json_string = metadata.dump();        // 한 줄로!
std::string pretty = metadata.dump(2);            // 들여쓰기 2칸
```

### 4. JSON 파싱

**jsoncpp:**
```cpp
Json::Value root;
Json::Reader reader;
bool success = reader.parse(json_string, root);
if (success) {
    std::string timestamp = root["timestamp"].asString();
}
```

**nlohmann/json:**
```cpp
try {
    json data = json::parse(json_string);
    std::string timestamp = data["timestamp"];  // 자동 타입 변환!
} catch (json::parse_error& e) {
    // 에러 처리
}
```

## 🎯 주요 장점 비교

### jsoncpp 장점:
- ✅ 오래된 프로젝트에서 많이 사용됨
- ✅ 안정성이 검증됨
- ✅ 명시적인 타입 변환 (`asString()`, `asInt()`)

### nlohmann/json 장점:
- ✅ **Header-only**: 컴파일 시 링킹 불필요
- ✅ **직관적인 문법**: STL 컨테이너처럼 사용 가능
- ✅ **간결한 코드**: 초기화 리스트 지원
- ✅ **자동 타입 변환**: 명시적 변환 불필요
- ✅ **모던 C++**: C++11/14/17 기능 활용
- ✅ **예외 기반 에러 처리**: try-catch로 안전함
- ✅ **더 나은 성능**: 대부분의 벤치마크에서 우수

## 🚀 실제 사용 예시

### nlohmann/json의 편리한 기능들:

```cpp
// 1. 초기화 리스트로 한 번에 생성
json person = {
    {"name", "John"},
    {"age", 30},
    {"hobbies", {"reading", "coding", "gaming"}}
};

// 2. STL 컨테이너처럼 사용
for (auto& hobby : person["hobbies"]) {
    std::cout << hobby << std::endl;
}

// 3. 자동 타입 변환
std::string name = person["name"];  // 자동으로 string으로 변환
int age = person["age"];           // 자동으로 int로 변환

// 4. 범위 기반 for문 지원
for (auto& [key, value] : person.items()) {
    std::cout << key << ": " << value << std::endl;
}
```

## 💡 추천 사항

**새로운 프로젝트라면 nlohmann/json 추천!**
- 더 간결하고 읽기 쉬운 코드
- Header-only로 의존성 관리 간편
- 모던 C++ 스타일
- 활발한 개발 및 커뮤니티

**기존 프로젝트에서 jsoncpp 사용 중이라면:**
- 이미 잘 작동하고 있다면 굳이 변경할 필요 없음
- 대규모 리팩토링이 필요한 경우에만 고려
