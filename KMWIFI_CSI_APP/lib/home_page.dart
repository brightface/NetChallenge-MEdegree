import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  final User user;

  HomePage(this.user);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text(
          '시각적 제약 없는 WIFI 행동예측',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        ),
      ),
      body: _buildBody()
    );
  }

  Widget _buildBody() {
    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: SafeArea(
        child: SingleChildScrollView(
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,

              children: <Widget>[
                Padding(padding: EdgeInsets.all(40.0)),
                Text(
                  'KM WIFI',
                  style: TextStyle(fontSize: 24.0),
                ),
                Padding(padding: EdgeInsets.all(8.0)),
                Text('행동인식 로그인됨'),
                Padding(padding: EdgeInsets.all(16.0)),
                SizedBox(
                  width: 260.0,
                  child: Card(
                    elevation: 4.0,
                    child: Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: <Widget>[
                          SizedBox(
                            width: 80.0,
                            height: 80.0,
                            child: CircleAvatar(
                              backgroundImage: NetworkImage(widget.user.photoURL),
                            ),
                          ),
                          Padding(padding: EdgeInsets.all(8.0)),
                          Text(
                            widget.user.email,
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Text(widget.user.displayName),
                          Padding(padding: EdgeInsets.all(8.0)),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: <Widget>[
                              SizedBox(
                                width: 70.0,
                                height: 70.0,
                                child: Image.network(
                                    'http://www.papernews1.com/news/photo/201911/1257_1175_2722.jpg',
                                    fit: BoxFit.cover),
                              ),
                              Padding(
                                padding: EdgeInsets.all(1.0),
                              ),
                              SizedBox(
                                width: 70.0,
                                height: 70.0,
                                child: Image.network(
                                    'https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAyMTAzMzFfMzcg%2FMDAxNjE3MTYzNDMzNDM0.xZy-gQOtLvC31fKLXs0hDTac6yo2OnmCKCdXaIrj9Hcg.WPzfzLuiYudnkChjD-33j0B2mOOe9RvbWPfilt5UDs8g.JPEG.petb84%2FIMG_2778.jpg&type=sc960_832',
                                    fit: BoxFit.cover),
                              ),
                              Padding(
                                padding: EdgeInsets.all(1.0),
                              ),
                              SizedBox(
                                width: 70.0,
                                height: 70.0,
                                child: Image.network(
                                    'https://search.pstatic.net/common/?src=http%3A%2F%2Fblogfiles.naver.net%2FMjAxNzA1MTZfMjk2%2FMDAxNDk0ODgwNzYyMzUy.fJ5meyVg30bkhDYm-z3ILBiSBEZHw-MAUB4N8yqWkvUg.lro1UG4eqgZTs6CGvHY9Ep1M6ctXZLTEdPYhkKy301Mg.JPEG.seunghwan08%2F%25BB%25FD%25C8%25C4_9%25B0%25B3%25BF%25F9_%25BE%25C6%25B1%25E2_%25BD%25BA%25BD%25BA%25B7%25CE_%25BE%25C9%25B1%25E2_%25BC%25BA%25B0%25F812.JPG&type=sc960_832',
                                    fit: BoxFit.cover),
                              ),
                            ],
                          ),
                          Padding(padding: EdgeInsets.all(4.0)),

                          Padding(padding: EdgeInsets.all(4.0)),

                        ],
                      ),
                    ),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
